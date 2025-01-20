from typing import Set


class Value:
    """Represents a scalar value and its gradient"""

    def __init__(
        self,
        data: float,
        _children: Set["Value"] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = data
        self.grad: float = 0.0
        self.backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: "Value") -> "Value":
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad = out.grad
            other.grad = out.grad

        out.backward = _backward
        return out

    def __mul__(self, other: "Value") -> "Value":
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out.backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0, self.data), (self,), "ReLU")

        def _backward() -> None:
            self.grad = out.grad * (self.data > 0)

        out.backward = _backward
        return out

    def sigmoid(self) -> "Value":
        out = Value(1 / (1 + math.exp(-self.data)), (self,), "Sigmoid")


