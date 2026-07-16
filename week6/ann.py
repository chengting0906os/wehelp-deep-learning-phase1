import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Activation Functions
# ---------------------------------------------------------------------------
class Activation(ABC):
    @abstractmethod
    def apply(self, *, values: list[float]) -> list[float]: ...

    @abstractmethod
    def derivative(self, *, values: list[float]) -> list[float]: ...


class Linear(Activation):
    def apply(self, *, values: list[float]) -> list[float]:
        return list(values)

    def derivative(self, *, values: list[float]) -> list[float]:
        return [1.0 for _ in values]


class ReLU(Activation):
    def apply(self, *, values: list[float]) -> list[float]:
        return [max(0.0, value) for value in values]

    def derivative(self, *, values: list[float]) -> list[float]:
        return [1.0 if value > 0 else 0.0 for value in values]


class Sigmoid(Activation):
    def apply(self, *, values: list[float]) -> list[float]:
        return [1 / (1 + math.exp(-value)) for value in values]

    def derivative(self, *, values: list[float]) -> list[float]:
        return [output * (1 - output) for output in self.apply(values=values)]


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------
class Loss(ABC):
    @abstractmethod
    def get_loss(self, *, outputs: list[float], expects: list[float]) -> float: ...

    @abstractmethod
    def get_output_gradients(
        self, *, outputs: list[float], expects: list[float]
    ) -> list[float]: ...


class MSE(Loss):
    def get_loss(self, *, outputs: list[float], expects: list[float]) -> float:
        n = len(outputs)
        return sum((e - o) ** 2 for o, e in zip(outputs, expects)) / n

    def get_output_gradients(
        self, *, outputs: list[float], expects: list[float]
    ) -> list[float]:
        n = len(outputs)
        return [2 / n * (o - e) for o, e in zip(outputs, expects)]


class BinaryCrossEntropy(Loss):
    def get_loss(self, *, outputs: list[float], expects: list[float]) -> float:
        return -sum(
            e * math.log(o) + (1 - e) * math.log(1 - o)
            for o, e in zip(outputs, expects)
        )

    def get_output_gradients(
        self, *, outputs: list[float], expects: list[float]
    ) -> list[float]:
        return [-e / o + (1 - e) / (1 - o) for o, e in zip(outputs, expects)]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class Neuron:
    def __init__(self, *, weights: list[float], bias: float) -> None:
        self.weights = weights
        self.bias = bias
        self.weight_gradients = [0.0] * len(weights)
        self.bias_gradient = 0.0


class Layer:
    def __init__(self, *, neurons: list[Neuron], activation: Activation) -> None:
        self.neurons = neurons
        self.activation = activation
        self.inputs: list[float] = []
        self.totals: list[float] = []

    def forward(self, *, inputs: list[float]) -> list[float]:
        self.inputs = list(inputs)
        self.totals = []

        for neuron in self.neurons:
            total = neuron.bias
            for input_value, weight in zip(inputs, neuron.weights):
                total += input_value * weight
            self.totals.append(total)

        return self.activation.apply(values=self.totals)

    def backward(self, *, output_gradients: list[float]) -> list[float]:
        """Compute this layer's weight/bias gradients from the incoming error, and return the error to pass back to the previous layer."""
        derivatives = self.activation.derivative(values=self.totals)
        deltas = [g * d for g, d in zip(output_gradients, derivatives)]

        for neuron, delta in zip(self.neurons, deltas):
            for index, input_value in enumerate(self.inputs):
                neuron.weight_gradients[index] += delta * input_value
            neuron.bias_gradient += delta

        input_gradients = [0.0] * len(self.inputs)
        for neuron, delta in zip(self.neurons, deltas):
            for index, weight in enumerate(neuron.weights):
                input_gradients[index] += delta * weight

        return input_gradients

    def zero_grad(self, *, learning_rate: float) -> None:
        """Update every weight and bias with `new = old - learning_rate * gradient`, then reset gradients to 0."""
        for neuron in self.neurons:
            neuron.weights = [
                w - learning_rate * g for w, g in zip(neuron.weights, neuron.weight_gradients)
            ]
            neuron.bias -= learning_rate * neuron.bias_gradient
            neuron.weight_gradients = [0.0] * len(neuron.weights)
            neuron.bias_gradient = 0.0


@dataclass(kw_only=True, frozen=True)
class Network:
    layers: list[Layer]

    def forward(self, *, inputs: list[float]) -> list[float]:
        values = inputs
        for layer in self.layers:
            values = layer.forward(inputs=values)
        return values

    def backward(self, *, output_gradients: list[float]) -> None:
        gradients = output_gradients
        for layer in reversed(self.layers):
            gradients = layer.backward(output_gradients=gradients)

    def zero_grad(self, *, learning_rate: float) -> None:
        for layer in self.layers:
            layer.zero_grad(learning_rate=learning_rate)

    def print_weights(self) -> None:
        for index, layer in enumerate(self.layers):
            print(f"Layer {index}")
            print([neuron.weights for neuron in layer.neurons])
            print([neuron.bias for neuron in layer.neurons])
