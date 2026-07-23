"""Week 6 Task 2 (variant): predict Titanic survival from sex, age and family size."""

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median, pstdev

from ann import BinaryCrossEntropy, Layer, Network, Neuron, ReLU, Sigmoid


@dataclass(frozen=True)
class Passenger:
    sex: float
    age: float | None
    family_size: float
    survived: float


@dataclass(frozen=True)
class Standardizer:
    mean: float
    standard_deviation: float

    def encode(self, value: float) -> float:
        return (value - self.mean) / self.standard_deviation


def load_data(data_path: Path) -> list[Passenger]:
    """Read Sex, Age and Survived from the assignment CSV."""
    passengers: list[Passenger] = []

    with data_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            # Male = 1, Female = 0
            sex = 1.0 if row["Sex"] == "male" else 0.0
            age = float(row["Age"]) if row["Age"] else None
            # Family size = siblings/spouses + parents/children + self
            family_size = float(int(row["SibSp"]) + int(row["Parch"]) + 1)
            passengers.append(
                Passenger(
                    sex=sex,
                    age=age,
                    family_size=family_size,
                    survived=float(row["Survived"]),
                )
            )

    return passengers


def encode_data(
    passengers: list[Passenger],
    missing_age: float,
    age_standardizer: Standardizer,
    family_standardizer: Standardizer,
) -> tuple[list[list[float]], list[list[float]]]:
    """Encode Sex, standardized Age and Family size as inputs, Survived as expects."""
    input_values: list[list[float]] = []
    expected_values: list[list[float]] = []

    for passenger in passengers:
        age = missing_age if passenger.age is None else passenger.age
        input_values.append(
            [
                passenger.sex,
                age_standardizer.encode(age),
                family_standardizer.encode(passenger.family_size),
            ]
        )
        expected_values.append([passenger.survived])

    return input_values, expected_values


def correct_rate(
    nn: Network,
    input_values: list[list[float]],
    expected_values: list[list[float]],
) -> float:
    """Return classification accuracy as a percentage."""
    correct_count = 0
    threshold = 0.5

    for input_value, expected_value in zip(input_values, expected_values):
        probability = nn.forward(inputs=input_value)[0]
        survival_status = 1.0 if probability >= threshold else 0.0
        if survival_status == expected_value[0]:
            correct_count += 1

    return correct_count / len(input_values) * 100


def main() -> None:
    # Set training parameters
    data_path = Path(__file__).parent / "data" / "titanic.csv"
    epochs = 500
    learning_rate = 0.001
    hidden_units = 8
    # 存活率 2 = 8 > 4
    # 存落率 0.001 = 0.01 > 0.005

    # Load data
    random_generator = random.Random(1)
    passengers = load_data(data_path)

    # Fill missing ages and standardize Age
    known_ages = [
        passenger.age for passenger in passengers if passenger.age is not None
    ]
    missing_age = median(known_ages)
    filled_ages = [
        missing_age if passenger.age is None else passenger.age
        for passenger in passengers
    ]
    age_standardizer = Standardizer(
        mean=fmean(filled_ages),
        standard_deviation=pstdev(filled_ages),
    )

    # Standardize Family size
    family_sizes = [passenger.family_size for passenger in passengers]
    family_standardizer = Standardizer(
        mean=fmean(family_sizes),
        standard_deviation=pstdev(family_sizes),
    )

    input_values, expected_values = encode_data(
        passengers, missing_age, age_standardizer, family_standardizer
    )

    # Build neural network (3 inputs -> ReLU hidden layer -> Sigmoid output)
    nn = Network(
        layers=[
            Layer(
                neurons=[
                    Neuron(
                        weights=[random_generator.uniform(-1.0, 1.0) for _ in range(3)],
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
                activation=Sigmoid(),
            ),
        ]
    )
    loss_function = BinaryCrossEntropy()
    order = list(range(len(input_values)))

    print("------ Task 2 (add family size) ------")

    # Evaluate the untrained network first, as a baseline
    print("------ Before Training ------")
    print(f"Correct Rate {correct_rate(nn, input_values, expected_values):.2f}%")

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

        # if (epoch + 1) % 20 == 0 or epoch == 0:
        #     average_loss = loss_sum / len(input_values)
        #     rate = correct_rate(nn, input_values, expected_values)
        #     print(
        #         f"Epoch {epoch + 1}: BCE = {average_loss:.5f}, "
        #         f"Correct rate = {rate:.2f}%"
        #     )

    # Final evaluation
    print("------ Start Evaluating ------")
    rate = correct_rate(nn, input_values, expected_values)
    print(f"Correct Rate {rate:.2f}%")


if __name__ == "__main__":
    main()
