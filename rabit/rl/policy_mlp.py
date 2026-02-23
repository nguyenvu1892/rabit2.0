from __future__ import annotations
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


class MLPPolicy:
    """
    1-hidden-layer MLP policy.

    Input: n_features
    Hidden: hidden_size (tanh)
    Output: 6 dims
      - logits dir: 3
      - tp_raw: 1
      - sl_raw: 1
      - hold_raw: 1
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 16,
        tp_range=(0.3, 1.2),
        sl_range=(0.3, 1.2),
        hold_range=(5, 60),
        seed: int = 42,
    ):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_out = 6

        self.tp_min, self.tp_max = tp_range
        self.sl_min, self.sl_max = sl_range
        self.hold_min, self.hold_max = hold_range

        rng = np.random.default_rng(seed)
        # Xavier-ish init
        self.W1 = rng.normal(0, 1.0 / np.sqrt(n_features), size=(hidden_size, n_features))
        self.b1 = np.zeros((hidden_size,), dtype=np.float64)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(hidden_size), size=(self.n_out, hidden_size))
        self.b2 = np.zeros((self.n_out,), dtype=np.float64)

    def get_params_flat(self) -> np.ndarray:
        return np.concatenate([
            self.W1.ravel(),
            self.b1,
            self.W2.ravel(),
            self.b2,
        ])

    def set_params_flat(self, theta: np.ndarray) -> None:
        p = 0
        w1_size = self.hidden_size * self.n_features
        self.W1 = theta[p:p + w1_size].reshape(self.hidden_size, self.n_features)
        p += w1_size

        self.b1 = theta[p:p + self.hidden_size]
        p += self.hidden_size

        w2_size = self.n_out * self.hidden_size
        self.W2 = theta[p:p + w2_size].reshape(self.n_out, self.hidden_size)
        p += w2_size

        self.b2 = theta[p:p + self.n_out]

    def act(self, x: np.ndarray) -> tuple[int, float, float, int]:
        h = np.tanh(self.W1 @ x + self.b1)          # (hidden,)
        y = self.W2 @ h + self.b2                   # (6,)

        logits = y[:3]
        probs = softmax(logits)
        dir_ = int(np.argmax(probs))  # 0 hold, 1 long, 2 short

        tp_u = np.tanh(y[3])
        sl_u = np.tanh(y[4])
        hold_u = np.tanh(y[5])

        tp_mult = self.tp_min + (tp_u + 1) * 0.5 * (self.tp_max - self.tp_min)
        sl_mult = self.sl_min + (sl_u + 1) * 0.5 * (self.sl_max - self.sl_min)
        hold_max = int(round(self.hold_min + (hold_u + 1) * 0.5 * (self.hold_max - self.hold_min)))

        return (dir_, float(tp_mult), float(sl_mult), int(hold_max))

    # Metadata for trainer compatibility
    @property
    def param_shapes(self):
        return {
            "n_features": self.n_features,
            "hidden_size": self.hidden_size,
            "n_out": self.n_out,
        }