"""Week 7 Task 2: Titanic survival classification with PyTorch (Dataset + DataLoader)."""

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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

    @classmethod
    def fit(cls, values: list[float]) -> "Standardizer":
        """Compute mean and standard deviation from the given (training-only) values."""
        tensor = torch.tensor(values, dtype=torch.float32)
        return cls(
            mean=float(tensor.mean()),
            standard_deviation=float(tensor.std()),
        )


def load_data(data_path: Path) -> list[Passenger]:
    """Read Sex, Age, family size and Survived from the assignment CSV."""
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


# ---------------------------------------------------------------------------
# Prepare Data with Dataset
# ---------------------------------------------------------------------------
class MyData(Dataset):
    def __init__(
        self,
        passengers: list[Passenger],
        missing_age: float,
        age_standardizer: Standardizer,
        family_standardizer: Standardizer,
    ) -> None:
        features: list[list[float]] = []
        targets: list[list[float]] = []

        for passenger in passengers:
            age = missing_age if passenger.age is None else passenger.age
            features.append(
                [
                    passenger.sex,
                    age_standardizer.encode(age),
                    family_standardizer.encode(passenger.family_size),
                ]
            )
            targets.append([passenger.survived])

        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.features)


class SurvivalClassifier(nn.Module):
    def __init__(self, hidden_units: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def correct_rate(model: nn.Module, loader: DataLoader) -> float:
    """Return classification accuracy (%) over the given DataLoader."""
    model.eval()
    correct_count = 0
    count = 0
    threshold = 0.5

    with torch.no_grad():
        for features, targets in loader:
            probabilities = model(features)
            predictions = (probabilities >= threshold).float()
            correct_count += (predictions == targets).sum().item()
            count += len(features)

    return correct_count / count * 100


def main() -> None:
    # Set training parameters
    data_path = Path(__file__).parent / "data" / "titanic.csv"
    epochs = 500
    learning_rate = 0.001
    batch_size = 64
    hidden_units = 8
    train_ratio = 0.8

    torch.manual_seed(1)
    generator = torch.Generator().manual_seed(1)

    # Load data and split into training / evaluate data in 4:1 ratio
    passengers = load_data(data_path)
    train_size = int(len(passengers) * train_ratio)
    shuffled = [
        passengers[index]
        for index in torch.randperm(len(passengers), generator=generator)
    ]
    train_passengers = shuffled[:train_size]
    eval_passengers = shuffled[train_size:]

    # Fit missing-age fill value and standardizers on training data only
    known_ages = [p.age for p in train_passengers if p.age is not None]
    missing_age = median(known_ages)
    filled_train_ages = [
        missing_age if p.age is None else p.age for p in train_passengers
    ]
    age_standardizer = Standardizer.fit(filled_train_ages)
    family_standardizer = Standardizer.fit(
        [p.family_size for p in train_passengers]
    )

    train_data = MyData(
        train_passengers, missing_age, age_standardizer, family_standardizer
    )
    eval_data = MyData(
        eval_passengers, missing_age, age_standardizer, family_standardizer
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)

    model = SurvivalClassifier(hidden_units=hidden_units)
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("------ Task 2 ------")

    # Evaluate the untrained network first, as a baseline
    print("------ Before Training ------")
    print(f"Correct Rate {correct_rate(model, eval_loader):.2f}%")

    # Training Procedure with Training Data
    print("------ Start Training ------")
    for epoch in range(epochs):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad()
            probabilities = model(features)
            loss = loss_function(probabilities, targets)
            loss.backward()
            optimizer.step()

    # Evaluating Procedure with Evaluate Data
    print("------ Start Evaluating ------")
    print(f"Correct Rate {correct_rate(model, eval_loader):.2f}%")


if __name__ == "__main__":
    main()
