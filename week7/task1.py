"""Week 7 Task 1: weight regression with PyTorch (Dataset + DataLoader)."""

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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

    @classmethod
    def fit(cls, values: list[float]) -> "Standardizer":
        """Compute mean and standard deviation from the given (training-only) values."""
        tensor = torch.tensor(values, dtype=torch.float32)
        return cls(
            mean=float(tensor.mean()),
            standard_deviation=float(tensor.std()),
        )


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


# ---------------------------------------------------------------------------
# Prepare Data with Dataset
# ---------------------------------------------------------------------------
class MyData(Dataset):
    def __init__(
        self,
        people: list[Person],
        height_standardizer: Standardizer,
        weight_standardizer: Standardizer,
    ) -> None:
        features = [
            [person.gender, height_standardizer.encode(person.height)]
            for person in people
        ]
        targets = [[weight_standardizer.encode(person.weight)] for person in people]
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.features)


class WeightRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Linear(2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def mean_absolute_error_in_pounds(
    model: nn.Module,
    loader: DataLoader,
    weight_standardizer: Standardizer,
) -> float:
    """Average absolute error (in pounds) over the given DataLoader."""
    model.eval()
    error_sum = 0.0
    count = 0

    with torch.no_grad():
        for features, targets in loader:
            predictions = model(features)
            predicted_weight = (
                predictions * weight_standardizer.standard_deviation
                + weight_standardizer.mean
            )
            actual_weight = (
                targets * weight_standardizer.standard_deviation
                + weight_standardizer.mean
            )
            error_sum += torch.abs(predicted_weight - actual_weight).sum().item()
            count += len(features)

    return error_sum / count


def main() -> None:
    # Set training parameters
    data_path = Path(__file__).parent / "data" / "weight-height.csv"
    epochs = 100
    learning_rate = 0.001
    batch_size = 64
    train_ratio = 0.8

    torch.manual_seed(1)
    generator = torch.Generator().manual_seed(1)

    # Load data and split into training / evaluate data in 4:1 ratio
    people = load_data(data_path)
    train_size = int(len(people) * train_ratio)
    shuffled = [people[index] for index in torch.randperm(len(people), generator=generator)]
    train_people = shuffled[:train_size]
    eval_people = shuffled[train_size:]

    # Standardize using training-data statistics only (avoid data leakage)
    height_standardizer = Standardizer.fit([person.height for person in train_people])
    weight_standardizer = Standardizer.fit([person.weight for person in train_people])

    train_data = MyData(train_people, height_standardizer, weight_standardizer)
    eval_data = MyData(eval_people, height_standardizer, weight_standardizer)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)

    model = WeightRegressor()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(
        f"Weight Average {weight_standardizer.mean} "
        f"Standard Deviation {weight_standardizer.standard_deviation}"
    )
    print("------ Task 1 ------")

    # Evaluate the untrained network first, as a baseline
    print("------ Before Training ------")
    evaluation_mae = mean_absolute_error_in_pounds(
        model,
        eval_loader,
        weight_standardizer,
    )
    print(f"Evaluation MAE: {evaluation_mae} pounds")

    # Training Procedure with Training Data
    print("------ Start Training ------")
    for epoch in range(epochs):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_function(predictions, targets)
            loss.backward()
            optimizer.step()

    # Evaluating Procedure with Evaluate Data
    print("------ Start Evaluating ------")
    evaluation_mae = mean_absolute_error_in_pounds(
        model,
        eval_loader,
        weight_standardizer,
    )
    print(f"Evaluation MAE: {evaluation_mae} pounds")


if __name__ == "__main__":
    main()
