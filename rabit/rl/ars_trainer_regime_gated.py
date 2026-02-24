from __future__ import annotations

import numpy as np

from rabit.env.metrics import compute_metrics
from rabit.env.trading_env_np import TradingEnvNP, make_np_window
from rabit.rl.confidence_gate import ConfidenceGateConfig, compute_confidence_gate


HOLD_ACTION = (0, 0.8, 0.8, 20)


def make_obs_matrix(df_feat, feature_cols: list[str]) -> np.ndarray:
    X = df_feat[feature_cols].to_numpy(dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-12
    X = (X - mu) / sd
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class ARSTrainerRegimeGated:
    """
    ARS windowed eval with robust reward + regime gating + confidence gate.

    If regime_arr[i] != target_regime => force HOLD (dir=0).
    If use_gate and gate disallows => force HOLD (dir=0).
    """

    def __init__(
        self,
        policy,
        sigma: float = 0.03,
        alpha: float = 0.02,
        n_directions: int = 10,
        top_k: int = 5,
        seed: int = 123,
        eval_windows: int = 8,
        window_size: int = 2000,
        seed_windows: int = 7,
        verbose: bool = False,
        dd_lambda: float = 0.25,
        trade_lambda: float = 0.05,
        dd_scale: float = 300.0,
        trade_scale: float = 250.0,
        target_regime: str = "trend",
        use_gate: bool = True,
        gate_cfg: ConfidenceGateConfig | None = None,
    ):
        self.policy = policy
        self.sigma = sigma
        self.alpha = alpha
        self.n_directions = n_directions
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)

        self.eval_windows = eval_windows
        self.window_size = window_size
        self.rng_win = np.random.default_rng(seed_windows)

        self.verbose = verbose

        self.dd_lambda = dd_lambda
        self.trade_lambda = trade_lambda
        self.dd_scale = dd_scale
        self.trade_scale = trade_scale

        self.target_regime = target_regime
        self.use_gate = use_gate
        self.gate_cfg = gate_cfg if gate_cfg is not None else ConfidenceGateConfig()

    def _robust_reward(self, pf: float, max_dd: float, trades: int) -> float:
        return float((pf - 1.0) - self.dd_lambda * (max_dd / (self.dd_scale + 1e-12)) - self.trade_lambda * (trades / (self.trade_scale + 1e-12)))

    def _slice_feat(self, feat_rows, start: int, end: int):
        if feat_rows is None:
            return None
        if hasattr(feat_rows, "iloc"):
            return feat_rows.iloc[start:end]
        return feat_rows[start:end]

    def _run_window(self, df_env, X, regime_arr: np.ndarray, feat_rows, start: int, end: int) -> float:
        if self.use_gate and feat_rows is None:
            raise ValueError("use_gate=True requires feat_rows")

        win = make_np_window(df_env, start, end)
        X_w = X[start:end]
        R_w = regime_arr[start:end]
        F_w = self._slice_feat(feat_rows, start, end)

        env = TradingEnvNP(
            win,
            gap_close_minutes=60,
            gap_skip_minutes=180,
            spread_open_cap=200,
            force_close_on_spread=False,
        )

        idx = {"i": 0}

        def policy_func(_):
            i = idx["i"]
            idx["i"] += 1

            # regime gating
            if R_w[i] != self.target_regime:
                return HOLD_ACTION

            # confidence gate
            if self.use_gate:
                row = F_w.iloc[i] if hasattr(F_w, "iloc") else F_w[i]
                _, allow, _ = compute_confidence_gate(row, self.gate_cfg)
                if not allow:
                    return HOLD_ACTION

            return self.policy.act(X_w[i])

        ledger = env.run_backtest(policy_func)
        m = compute_metrics(ledger)
        return self._robust_reward(m.profit_factor, m.max_drawdown, m.trades)

    def evaluate(self, df_env, X, regime_arr: np.ndarray, feat_rows) -> float:
        n = len(df_env)
        if n <= self.window_size + 10:
            return self._run_window(df_env, X, regime_arr, feat_rows, 0, n)

        rewards = []
        for _ in range(self.eval_windows):
            start = int(self.rng_win.integers(0, n - self.window_size))
            end = start + self.window_size
            rewards.append(self._run_window(df_env, X, regime_arr, feat_rows, start, end))

        return float(np.mean(rewards))

    def train(self, df_env, X, regime_arr: np.ndarray, feat_rows, iters: int = 15):
        theta = self.policy.get_params_flat()

        self.policy.set_params_flat(theta)
        base_r = self.evaluate(df_env, X, regime_arr, feat_rows)
        print({"regime": self.target_regime, "iter": -1, "reward": base_r, "note": "baseline (regime+gate)"})

        best_theta = theta.copy()
        best_reward = float(base_r)
        history = []

        for it in range(iters):
            deltas = self.rng.normal(0, 1, size=(self.n_directions, theta.size))
            rewards_pos = np.zeros((self.n_directions,), dtype=np.float64)
            rewards_neg = np.zeros((self.n_directions,), dtype=np.float64)

            for k in range(self.n_directions):
                if self.verbose:
                    print(f"[ARS-Regime-Gated] regime={self.target_regime} iter={it} dir={k+1}/{self.n_directions}")

                self.policy.set_params_flat(theta + self.sigma * deltas[k])
                rewards_pos[k] = self.evaluate(df_env, X, regime_arr, feat_rows)

                self.policy.set_params_flat(theta - self.sigma * deltas[k])
                rewards_neg[k] = self.evaluate(df_env, X, regime_arr, feat_rows)

            scores = np.maximum(rewards_pos, rewards_neg)
            top_idx = np.argsort(scores)[-self.top_k:]

            r_std = float(np.std(np.concatenate([rewards_pos[top_idx], rewards_neg[top_idx]])) + 1e-12)

            step = np.zeros_like(theta)
            for k in top_idx:
                step += (rewards_pos[k] - rewards_neg[k]) * deltas[k]

            theta = theta + (self.alpha / (self.top_k * r_std)) * step

            self.policy.set_params_flat(theta)
            r = float(self.evaluate(df_env, X, regime_arr, feat_rows))

            if r > best_reward:
                best_reward = r
                best_theta = theta.copy()

            rec = {
                "regime": self.target_regime,
                "iter": it,
                "reward": float(r),
                "best_reward": float(best_reward),
                "r_std": float(r_std),
                "mean_pos": float(np.mean(rewards_pos)),
                "mean_neg": float(np.mean(rewards_neg)),
            }
            history.append(rec)
            print(rec)

        self.policy.set_params_flat(best_theta)
        return history, best_theta, best_reward
