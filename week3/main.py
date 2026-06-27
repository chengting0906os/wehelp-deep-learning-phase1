from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Layer:
    weights: list[list[float]]
    biases: list[float]


@dataclass(frozen=True, kw_only=True)
class Network:
    layers: list[Layer]

    def forward(self, inputs: list[float]) -> list[float]:
        values = inputs

        for layer in self.layers:
            next_values = []

            for neuron_weights, bias in zip(layer.weights, layer.biases):
                total = 0

                for value, weight in zip(values, neuron_weights):
                    total += value * weight

                output = total + bias
                next_values.append(output)

            values = next_values

        return values


def run_task1() -> None:
    nn = Network(
        layers=[
            Layer(
                weights=[
                    [0.5, 0.2],
                    [0.6, -0.6],
                ],
                biases=[0.3, 0.25],
            ),
            Layer(
                weights=[
                    [0.8, 0.4],
                ],
                biases=[-0.5],
            ),
        ]
    )

    outputs = nn.forward([1.5, 0.5])
    print(outputs)

    outputs = nn.forward([0, 1])
    print(outputs)


def run_task2() -> None:
    nn = Network(
        layers=[
            Layer(
                weights=[
                    [0.5, 1.5],
                    [0.6, -0.8],
                ],
                biases=[0.3, 1.25],
            ),
            Layer(
                weights=[
                    [0.6, -0.8],
                ],
                biases=[0.3],
            ),
            Layer(
                weights=[
                    [0.5],
                    [-0.4],
                ],
                biases=[0.2, 0.5],
            ),
        ]
    )
    
    outputs = nn.forward([0.75, 1.25])
    print(outputs)
    
    outputs = nn.forward([-1, 0.5])
    print(outputs)
    
if __name__ == "__main__":
    run_task1()
    print()
    run_task2()
