from __future__ import annotations

import numpy as np

from rabit.env.metrics import compute_metrics
from rabit.env.trading_env_np import TradingEnvNP, make_np_window


def make_obs_matrix(df_feat, feature_cols: list[str]) -> np.ndarray:
    X = df_feat[feature_cols].to_numpy(dtype=np.float64)

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-12
    X = (X - mu) / sd

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


class ARSTrainer:
    """
    ARS trainer (windowed evaluation) using numpy-optimized env for speed.
    Robust reward:
      reward = (PF - 1) - dd_lambda*(maxDD/dd_scale) - trade_lambda*(trades/trade_scale)
    """

    def __init__(
        self,
        policy,
        sigma: float = 0.03,
        alpha: float = 0.02,
        n_directions: int = 12,
        top_k: int = 6,
        seed: int = 123,
        eval_windows: int = 8,
        window_size: int = 2000,
        seed_windows: int = 7,
        verbose: bool = True,
        dd_lambda: float = 0.25,
        trade_lambda: float = 0.05,
        dd_scale: float = 300.0,
        trade_scale: float = 250.0,
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

    def _robust_reward(self, pf: float, max_dd: float, trades: int) -> float:
        pf_term = pf - 1.0
        dd_term = self.dd_lambda * (max_dd / (self.dd_scale + 1e-12))
        tr_term = self.trade_lambda * (trades / (self.trade_scale + 1e-12))
        return float(pf_term - dd_term - tr_term)

    def _run_window(self, df_env, X, start: int, end: int) -> float:
        # Build NP window (fast arrays + precomputed gaps)
        win = make_np_window(df_env, start, end)
        X_w = X[start:end]

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
            return self.policy.act(X_w[i])

        ledger = env.run_backtest(policy_func)
        m = compute_metrics(ledger)
        return self._robust_reward(m.profit_factor, m.max_drawdown, m.trades)

    def evaluate(self, df_env, X) -> float:
        n = len(df_env)
        if n <= self.window_size + 10:
            return self._run_window(df_env, X, 0, n)

        rewards = []
        for _ in range(self.eval_windows):
            start = int(self.rng_win.integers(0, n - self.window_size))
            end = start + self.window_size
            rewards.append(self._run_window(df_env, X, start, end))

        return float(np.mean(rewards))

    def train(self, df_env, X, iters: int = 20):
        theta = self.policy.get_params_flat()

        self.policy.set_params_flat(theta)
        base_r = self.evaluate(df_env, X)
        print({"iter": -1, "reward": base_r, "note": "baseline theta (robust, np-env)"})

        best_theta = theta.copy()
        best_reward = float(base_r)

        history = []

        for it in range(iters):
            deltas = self.rng.normal(0, 1, size=(self.n_directions, theta.size))
            rewards_pos = np.zeros((self.n_directions,), dtype=np.float64)
            rewards_neg = np.zeros((self.n_directions,), dtype=np.float64)

            for k in range(self.n_directions):
                if self.verbose:
                    print(f"[ARS] iter={it} dir={k+1}/{self.n_directions}")

                self.policy.set_params_flat(theta + self.sigma * deltas[k])
                rewards_pos[k] = self.evaluate(df_env, X)

                self.policy.set_params_flat(theta - self.sigma * deltas[k])
                rewards_neg[k] = self.evaluate(df_env, X)

            scores = np.maximum(rewards_pos, rewards_neg)
            top_idx = np.argsort(scores)[-self.top_k:]

            r_std = float(np.std(np.concatenate([rewards_pos[top_idx], rewards_neg[top_idx]])) + 1e-12)

            step = np.zeros_like(theta)
            for k in top_idx:
                step += (rewards_pos[k] - rewards_neg[k]) * deltas[k]

            theta = theta + (self.alpha / (self.top_k * r_std)) * step

            self.policy.set_params_flat(theta)
            r = float(self.evaluate(df_env, X))

            if r > best_reward:
                best_reward = r
                best_theta = theta.copy()

            rec = {
                "iter": it,
                "reward": r,
                "best_reward": best_reward,
                "r_std": float(r_std),
                "best_dir_reward": float(np.max(scores)),
                "mean_pos": float(np.mean(rewards_pos)),
                "mean_neg": float(np.mean(rewards_neg)),
            }
            history.append(rec)
            print(rec)

        return history, best_theta, best_reward