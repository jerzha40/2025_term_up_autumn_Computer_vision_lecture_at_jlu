import numpy as np
from . import _fastnn


class Layer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, in_features: int, out_features: int):
        self.weights = (
            np.random.randn(in_features, out_features).astype(np.float32) * 0.01
        )
        self.bias = np.zeros((1, out_features), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x.astype(np.float32)
        self.output = _fastnn.dense_forward(self.input, self.weights, self.bias)
        return self.output

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        # 可以先用 numpy 写，后续替换成 _fastnn.dense_backward
        dW = self.input.T @ grad
        dB = grad.sum(axis=0, keepdims=True)
        dX = grad @ self.weights.T

        # 更新
        self.weights -= lr * dW
        self.bias -= lr * dB
        return dX
