from __future__ import annotations

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from rabit.env.metrics import compute_metrics
from rabit.env.trading_env_np import TradingEnvNP, make_np_window


def make_obs_matrix(df_feat, feature_cols: list[str]) -> np.ndarray:
    X = df_feat[feature_cols].to_numpy(dtype=np.float64)

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-12
    X = (X - mu) / sd

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def _robust_reward(pf: float, max_dd: float, trades: int, dd_lambda: float, trade_lambda: float, dd_scale: float, trade_scale: float) -> float:
    pf_term = pf - 1.0
    dd_term = dd_lambda * (max_dd / (dd_scale + 1e-12))
    tr_term = trade_lambda * (trades / (trade_scale + 1e-12))
    return float(pf_term - dd_term - tr_term)


def _eval_theta_on_windows(
    theta: np.ndarray,
    W_shape: tuple[int, int],
    b_len: int,
    X: np.ndarray,
    df_env,  # pandas df slice (picklable)
    windows: list[tuple[int, int]],
    tp_min: float,
    tp_max: float,
    sl_min: float,
    sl_max: float,
    hold_min: int,
    hold_max: int,
    dd_lambda: float,
    trade_lambda: float,
    dd_scale: float,
    trade_scale: float,
) -> float:
    """
    Evaluate one theta on multiple windows and return mean robust reward.
    Re-implements LinearPolicy.act inline to avoid importing user objects.
    """
    n_out, n_feat = W_shape
    w_size = n_out * n_feat
    W = theta[:w_size].reshape(n_out, n_feat)
    b = theta[w_size:w_size + b_len]

    def act(x: np.ndarray):
        y = W @ x + b  # (6,)
        logits = y[:3]
        logits = logits - np.max(logits)
        e = np.exp(logits)
        probs = e / (np.sum(e) + 1e-12)
        dir_ = int(np.argmax(probs))

        tp_u = np.tanh(y[3])
        sl_u = np.tanh(y[4])
        hold_u = np.tanh(y[5])

        tp_mult = tp_min + (tp_u + 1) * 0.5 * (tp_max - tp_min)
        sl_mult = sl_min + (sl_u + 1) * 0.5 * (sl_max - sl_min)
        h = int(round(hold_min + (hold_u + 1) * 0.5 * (hold_max - hold_min)))

        return dir_, float(tp_mult), float(sl_mult), int(h)

    rewards = []
    for (start, end) in windows:
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
            return act(X_w[i])

        ledger = env.run_backtest(policy_func)
        m = compute_metrics(ledger)
        rewards.append(_robust_reward(m.profit_factor, m.max_drawdown, m.trades, dd_lambda, trade_lambda, dd_scale, trade_scale))

    return float(np.mean(rewards)) if rewards else 0.0


class ARSParallelTrainer:
    """
    Parallel ARS using ProcessPoolExecutor.
    - Parallelizes evaluation of (theta + noise) and (theta - noise) across directions
    - Keeps window list fixed per 'evaluate' call to reduce randomness and overhead
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
        max_workers: int | None = None,
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

        self.max_workers = max_workers or max(1, (os.cpu_count() or 4) - 1)

        self.dd_lambda = dd_lambda
        self.trade_lambda = trade_lambda
        self.dd_scale = dd_scale
        self.trade_scale = trade_scale

        # policy param shapes
        self.n_out = 6
        self.n_feat = policy.n_features
        self.b_len = self.n_out

        # action bounds from policy
        self.tp_min, self.tp_max = policy.tp_min, policy.tp_max
        self.sl_min, self.sl_max = policy.sl_min, policy.sl_max
        self.hold_min, self.hold_max = policy.hold_min, policy.hold_max

    def _sample_windows(self, n: int) -> list[tuple[int, int]]:
        if n <= self.window_size + 10:
            return [(0, n)]
        windows = []
        for _ in range(self.eval_windows):
            start = int(self.rng_win.integers(0, n - self.window_size))
            windows.append((start, start + self.window_size))
        return windows

    def evaluate(self, df_env, X, theta: np.ndarray) -> float:
        windows = self._sample_windows(len(df_env))
        return _eval_theta_on_windows(
            theta=theta,
            W_shape=(self.n_out, self.n_feat),
            b_len=self.b_len,
            X=X,
            df_env=df_env,
            windows=windows,
            tp_min=self.tp_min, tp_max=self.tp_max,
            sl_min=self.sl_min, sl_max=self.sl_max,
            hold_min=self.hold_min, hold_max=self.hold_max,
            dd_lambda=self.dd_lambda,
            trade_lambda=self.trade_lambda,
            dd_scale=self.dd_scale,
            trade_scale=self.trade_scale,
        )

    def train(self, df_env, X, iters: int = 20):
        theta = self.policy.get_params_flat()

        base_r = self.evaluate(df_env, X, theta)
        print({"iter": -1, "reward": base_r, "note": "baseline theta (parallel)"})

        best_theta = theta.copy()
        best_reward = float(base_r)

        history = []

        for it in range(iters):
            deltas = self.rng.normal(0, 1, size=(self.n_directions, theta.size))

            # pre-sample ONE set of windows for ALL evaluations in this iter (reduces noise, speeds a bit)
            windows = self._sample_windows(len(df_env))

            # schedule all tasks
            futures = {}
            with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
                for k in range(self.n_directions):
                    th_pos = theta + self.sigma * deltas[k]
                    th_neg = theta - self.sigma * deltas[k]

                    fpos = ex.submit(
                        _eval_theta_on_windows,
                        th_pos,
                        (self.n_out, self.n_feat),
                        self.b_len,
                        X,
                        df_env,
                        windows,
                        self.tp_min, self.tp_max,
                        self.sl_min, self.sl_max,
                        self.hold_min, self.hold_max,
                        self.dd_lambda, self.trade_lambda, self.dd_scale, self.trade_scale,
                    )
                    futures[fpos] = ("pos", k)

                    fneg = ex.submit(
                        _eval_theta_on_windows,
                        th_neg,
                        (self.n_out, self.n_feat),
                        self.b_len,
                        X,
                        df_env,
                        windows,
                        self.tp_min, self.tp_max,
                        self.sl_min, self.sl_max,
                        self.hold_min, self.hold_max,
                        self.dd_lambda, self.trade_lambda, self.dd_scale, self.trade_scale,
                    )
                    futures[fneg] = ("neg", k)

                rewards_pos = np.zeros((self.n_directions,), dtype=np.float64)
                rewards_neg = np.zeros((self.n_directions,), dtype=np.float64)

                for fut in as_completed(futures):
                    kind, k = futures[fut]
                    r = float(fut.result())
                    if kind == "pos":
                        rewards_pos[k] = r
                    else:
                        rewards_neg[k] = r

            scores = np.maximum(rewards_pos, rewards_neg)
            top_idx = np.argsort(scores)[-self.top_k:]

            r_std = float(np.std(np.concatenate([rewards_pos[top_idx], rewards_neg[top_idx]])) + 1e-12)

            step = np.zeros_like(theta)
            for k in top_idx:
                step += (rewards_pos[k] - rewards_neg[k]) * deltas[k]

            theta = theta + (self.alpha / (self.top_k * r_std)) * step

            r = self.evaluate(df_env, X, theta)

            if r > best_reward:
                best_reward = r
                best_theta = theta.copy()

            rec = {
                "iter": it,
                "reward": float(r),
                "best_reward": float(best_reward),
                "r_std": float(r_std),
                "best_dir_reward": float(np.max(scores)),
                "mean_pos": float(np.mean(rewards_pos)),
                "mean_neg": float(np.mean(rewards_neg)),
                "workers": int(self.max_workers),
            }
            history.append(rec)
            print(rec)

        # sync policy object with best theta
        self.policy.set_params_flat(best_theta)
        return history, best_theta, best_reward