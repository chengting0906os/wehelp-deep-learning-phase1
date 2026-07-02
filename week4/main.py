import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Activation Functions
# ---------------------------------------------------------------------------
class Activation(ABC):
    @abstractmethod
    def apply(self, values: list[float]) -> list[float]: ...


class Linear(Activation):
    def apply(self, values: list[float]) -> list[float]:
        return list(values)


class ReLU(Activation):
    def apply(self, values: list[float]) -> list[float]:
        return [max(0.0, value) for value in values]


class Sigmoid(Activation):
    def apply(self, values: list[float]) -> list[float]:
        return [1 / (1 + math.exp(-value)) for value in values]


class Softmax(Activation):
    def apply(self, values: list[float]) -> list[float]:
        shift = max(values)
        exps = [math.exp(value - shift) for value in values]
        total = sum(exps)
        return [exp / total for exp in exps]


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
class Loss(ABC):
    @abstractmethod
    def total(self, outputs: list[float], expects: list[float]) -> float: ...

    def __call__(self, outputs: list[float], expects: list[float]) -> float:
        return self.total(outputs, expects)


class MSE(Loss):
    def total(self, outputs: list[float], expects: list[float]) -> float:
        n = len(outputs)
        return sum((e - o) ** 2 for o, e in zip(outputs, expects)) / n


class BinaryCrossEntropy(Loss):
    def total(self, outputs: list[float], expects: list[float]) -> float:
        return -sum(
            e * math.log(o) + (1 - e) * math.log(1 - o)
            for o, e in zip(outputs, expects)
        )


class CategoricalCrossEntropy(Loss):
    def total(self, outputs: list[float], expects: list[float]) -> float:
        return -sum(e * math.log(o) for o, e in zip(outputs, expects))


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class Neuron:
    weights: list[float]
    bias: float


@dataclass(frozen=True, kw_only=True)
class Layer:
    neurons: list[Neuron]
    activation: Activation


@dataclass(frozen=True, kw_only=True)
class Network:
    layers: list[Layer]

    def forward(self, inputs: list[float]) -> list[float]:
        values = inputs

        for layer in self.layers:
            totals = []

            for neuron in layer.neurons:
                total = neuron.bias
                for value, weight in zip(values, neuron.weights):
                    total += value * weight
                totals.append(total)

            values = layer.activation.apply(totals)

        return values


# ---------------------------------------------------------------------------
# Task 1 - Regression (ReLU hidden, Linear output, MSE loss)
# ---------------------------------------------------------------------------
def run_regression() -> None:
    nn = Network(
        layers=[
            Layer(
                neurons=[
                    Neuron(weights=[0.5, 0.2], bias=0.3),
                    Neuron(weights=[0.6, -0.6], bias=0.25),
                ],
                activation=ReLU(),
            ),
            Layer(
                neurons=[
                    Neuron(weights=[0.8, -0.5], bias=0.6),
                    Neuron(weights=[0.4, 0.5], bias=-0.25),
                ],
                activation=Linear(),
            ),
        ]
    )
    loss = MSE()

    outputs = nn.forward([1.5, 0.5])
    print("Outputs", outputs)
    print("Total Loss", loss(outputs, [0.8, 1]))

    outputs = nn.forward([0, 1])
    print("Outputs", outputs)
    print("Total Loss", loss(outputs, [0.5, 0.5]))


# ---------------------------------------------------------------------------
# Task 2 - Binary Classification (ReLU hidden, Sigmoid output, BCE loss)
# ---------------------------------------------------------------------------
def run_binary_classification() -> None:
    nn = Network(
        layers=[
            Layer(
                neurons=[
                    Neuron(weights=[0.5, 0.2], bias=0.3),
                    Neuron(weights=[0.6, -0.6], bias=0.25),
                ],
                activation=ReLU(),
            ),
            Layer(
                neurons=[
                    Neuron(weights=[0.8, 0.4], bias=-0.5),
                ],
                activation=Sigmoid(),
            ),
        ]
    )
    loss = BinaryCrossEntropy()

    outputs = nn.forward([0.75, 1.25])
    print("Outputs", outputs)
    print("Total Loss", loss(outputs, [1]))

    outputs = nn.forward([-1, 0.5])
    print("Outputs", outputs)
    print("Total Loss", loss(outputs, [0]))


# ---------------------------------------------------------------------------
# Task 3 - Multi-Label Classification (ReLU hidden, Sigmoid output, BCE loss)
# ---------------------------------------------------------------------------
def run_multi_label_classification() -> None:
    nn = Network(
        layers=[
            Layer(
                neurons=[
                    Neuron(weights=[0.5, 0.2], bias=0.3),
                    Neuron(weights=[0.6, -0.6], bias=0.25),
                ],
                activation=ReLU(),
            ),
            Layer(
                neurons=[
                    Neuron(weights=[0.8, -0.4], bias=0.6),
                    Neuron(weights=[0.5, 0.4], bias=0.5),
                    Neuron(weights=[0.3, 0.75], bias=-0.5),
                ],
                activation=Sigmoid(),
            ),
        ]
    )
    loss = BinaryCrossEntropy()

    outputs = nn.forward([1.5, 0.5])
    print("Outputs", outputs)
    print("Total Loss", loss(outputs, [1, 0, 1]))

    outputs = nn.forward([0, 1])
    print("Outputs", outputs)
    print("Total Loss", loss(outputs, [1, 1, 0]))


# ---------------------------------------------------------------------------
# Task 4 - Multi-Class Classification (ReLU hidden, Softmax output, CCE loss)
# ---------------------------------------------------------------------------
def run_multi_class_classification() -> None:
    nn = Network(
        layers=[
            Layer(
                neurons=[
                    Neuron(weights=[0.5, 0.2], bias=0.3),
                    Neuron(weights=[0.6, -0.6], bias=0.25),
                ],
                activation=ReLU(),
            ),
            Layer(
                neurons=[
                    Neuron(weights=[0.8, -0.4], bias=0.6),
                    Neuron(weights=[0.5, 0.4], bias=0.5),
                    Neuron(weights=[0.3, 0.75], bias=-0.5),
                ],
                activation=Softmax(),
            ),
        ]
    )
    loss = CategoricalCrossEntropy()

    outputs = nn.forward([1.5, 0.5])
    print("Outputs", outputs)
    print("Total Loss", loss(outputs, [1, 0, 0]))

    outputs = nn.forward([0, 1])
    print("Outputs", outputs)
    print("Total Loss", loss(outputs, [0, 0, 1]))


if __name__ == "__main__":
    print("------ Regression ------")
    run_regression()
    print()

    print("------ Binary Classification ------")
    run_binary_classification()
    print()

    print("------ Multi-Label Classification ------")
    run_multi_label_classification()
    print()

    print("------ Multi-Class Classification ------")
    run_multi_class_classification()
