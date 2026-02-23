from __future__ import annotations
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


class LinearPolicy:
    """
    Linear policy over features.

    Outputs:
      - dir logits: 3 (hold, long, short)
      - tp_raw: 1
      - sl_raw: 1
      - hold_raw: 1

    Total outputs = 6
    """

    def __init__(
        self,
        n_features: int,
        tp_range=(0.3, 1.2),
        sl_range=(0.3, 1.2),
        hold_range=(5, 60),
        seed: int = 42,
    ):
        self.n_features = n_features
        self.n_out = 6
        self.tp_min, self.tp_max = tp_range
        self.sl_min, self.sl_max = sl_range
        self.hold_min, self.hold_max = hold_range

        rng = np.random.default_rng(seed)
        # weights: (n_out, n_features) and bias: (n_out,)
        self.W = rng.normal(0, 0.01, size=(self.n_out, n_features))
        self.b = np.zeros((self.n_out,), dtype=np.float64)

    def get_params_flat(self) -> np.ndarray:
        return np.concatenate([self.W.flatten(), self.b])

    def set_params_flat(self, theta: np.ndarray) -> None:
        w_size = self.n_out * self.n_features
        self.W = theta[:w_size].reshape(self.n_out, self.n_features)
        self.b = theta[w_size:w_size + self.n_out]

    def act(self, x: np.ndarray) -> tuple[int, float, float, int]:
        """
        x: feature vector shape (n_features,)
        """
        y = self.W @ x + self.b  # (6,)
        logits = y[:3]
        probs = softmax(logits)
        dir_ = int(np.argmax(probs))  # 0 hold, 1 long, 2 short

        # bounded continuous outputs via tanh
        tp_u = np.tanh(y[3])  # [-1,1]
        sl_u = np.tanh(y[4])
        hold_u = np.tanh(y[5])

        tp_mult = self.tp_min + (tp_u + 1) * 0.5 * (self.tp_max - self.tp_min)
        sl_mult = self.sl_min + (sl_u + 1) * 0.5 * (self.sl_max - self.sl_min)
        hold_max = int(round(self.hold_min + (hold_u + 1) * 0.5 * (self.hold_max - self.hold_min)))

        return (dir_, float(tp_mult), float(sl_mult), int(hold_max))