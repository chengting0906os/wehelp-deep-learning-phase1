"""Week 6 Task 1: predict weight from gender and height."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev

from ann import MSE, Layer, Linear, Network, Neuron, ReLU


@dataclass(frozen=True)
class Person:
    gender: float
    height: float
    weight: float


@dataclass(frozen=True)
class Standardizer:
    mean: float
    standard_deviation: float

    def encode(self, value: float) -> float:
        return (value - self.mean) / self.standard_deviation

    def decode(self, value: float) -> float:
        return value * self.standard_deviation + self.mean


def load_data(data_path: Path) -> list[Person]:
    """Read Gender, Height and Weight from the assignment CSV."""
    people: list[Person] = []

    with data_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            # Male = 1, Female = 0
            gender = 1.0 if row["Gender"] == "Male" else 0.0
            people.append(
                Person(
                    gender=gender,
                    height=float(row["Height"]),
                    weight=float(row["Weight"]),
                )
            )

    return people


def average_error_in_pounds(
    nn: Network,
    input_values: list[list[float]],
    actual_weights: list[float],
    weight_standardizer: Standardizer,
) -> float:
    """Average absolute error (in pounds) over the whole dataset."""
    error_sum = 0.0
    for input_value, actual_weight in zip(input_values, actual_weights):
        predicted_weight = weight_standardizer.decode(nn.forward(inputs=input_value)[0])
        error_sum += abs(predicted_weight - actual_weight)
    return error_sum / len(input_values)


def main() -> None:
    # Set training parameters
    data_path = Path(__file__).parent / "data" / "weight-height.csv"
    epochs = 100
    learning_rate = 0.0001
    hidden_units = 4

    # 誤差 4<16=8<2
    # 誤差 0.0001(7.99) < 0.001(8.05 下不去)

    # Load and encode data
    random_generator = random.Random(1)
    people = load_data(data_path)
    input_values = [[person.gender, person.height] for person in people]
    actual_weights = [person.weight for person in people]

    # Standardize Height and Weight
    heights = [input_value[1] for input_value in input_values]
    height_standardizer = Standardizer(
        mean=fmean(heights),
        standard_deviation=pstdev(heights),
    )
    encoded_heights = [height_standardizer.encode(height) for height in heights]

    weight_standardizer = Standardizer(
        mean=fmean(actual_weights),
        standard_deviation=pstdev(actual_weights),
    )
    encoded_weights = [weight_standardizer.encode(weight) for weight in actual_weights]
    input_values = [
        [input_value[0], height]
        for input_value, height in zip(input_values, encoded_heights)
    ]
    expected_values = [[weight] for weight in encoded_weights]

    # Build neural network (weights start as small random numbers between -1 and 1)
    nn = Network(
        layers=[
            Layer(
                neurons=[
                    Neuron(
                        weights=[random_generator.uniform(-1.0, 1.0) for _ in range(2)],
                        bias=0.0,
                    )
                    for _ in range(hidden_units)
                ],
                activation=ReLU(),
            ),
            Layer(
                neurons=[
                    Neuron(
                        weights=[
                            random_generator.uniform(-1.0, 1.0)
                            for _ in range(hidden_units)
                        ],
                        bias=0.0,
                    )
                ],
                activation=Linear(),
            ),
        ]
    )
    loss_function = MSE()
    order = list(range(len(input_values)))

    # Show the dataset's weight statistics
    print(
        f"Weight Average {weight_standardizer.mean} "
        f"Standard Deviation {weight_standardizer.standard_deviation}"
    )
    print("------ Task 1 ------")

    # Evaluate the untrained (random-weight) network first, as a baseline
    print("------ Before Training ------")
    average_error = average_error_in_pounds(
        nn, input_values, actual_weights, weight_standardizer
    )
    print(f"Average Loss in Weight {average_error}")

    # Train neural network
    print("------ Start Training ------")
    for epoch in range(epochs):
        loss_sum = 0.0

        for index in order:
            outputs = nn.forward(inputs=input_values[index])
            loss_sum += loss_function.get_loss(
                outputs=outputs, expects=expected_values[index]
            )
            output_gradients = loss_function.get_output_gradients(
                outputs=outputs, expects=expected_values[index]
            )
            nn.backward(output_gradients=output_gradients)
            nn.zero_grad(learning_rate=learning_rate)

        # if (epoch + 1) % 10 == 0 or epoch == 0:
        #     average_loss = loss_sum / len(input_values)
        #     average_error = average_error_in_pounds(
        #         nn, input_values, actual_weights, weight_standardizer
        #     )
        #     print(
        #         f"Epoch {epoch + 1}: MSE = {average_loss:.5f}, "
        #         f"Average error = {average_error:.2f} lb"
        #     )

    # Final evaluation
    print("------ Start Evaluating ------")
    average_error = average_error_in_pounds(
        nn, input_values, actual_weights, weight_standardizer
    )
    print(f"Average Loss in Weight {average_error}")


if __name__ == "__main__":
    main()
