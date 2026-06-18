import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Task 1: points, lines, circles, polygons
# ---------------------------------------------------------------------------
@dataclass(frozen=True, kw_only=True)
class Point:
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True, kw_only=True)
class Line:
    start: Point
    end: Point

    @property
    def vector(self) -> tuple[float, float]:
        return (self.end.x - self.start.x, self.end.y - self.start.y)

    def cross(self, other: "Line") -> float:
        ax, ay = self.vector
        bx, by = other.vector
        return ax * by - ay * bx

    def dot(self, other: "Line") -> float:
        ax, ay = self.vector
        bx, by = other.vector
        return ax * bx + ay * by

    def is_parallel(self, other: "Line") -> bool:
        return self.cross(other) == 0

    def is_perpendicular(self, other: "Line") -> bool:
        return self.dot(other) == 0


@dataclass(frozen=True, kw_only=True)
class Circle:
    center: Point
    radius: float

    def area(self) -> float:
        return math.pi * self.radius**2

    def intersects(self, other: "Circle") -> bool:
        distance = self.center.distance_to(other.center)
        return abs(self.radius - other.radius) <= distance <= self.radius + other.radius


@dataclass(frozen=True, kw_only=True)
class Polygon:
    vertices: list[Point]

    def perimeter(self) -> float:
        total = 0.0
        for i in range(len(self.vertices)):
            current = self.vertices[i]
            nxt = self.vertices[(i + 1) % len(self.vertices)]
            total += current.distance_to(nxt)
        return total


# ---------------------------------------------------------------------------
# Task 2: tower defense units
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class Enemy:
    label: str
    x: float
    y: float
    vx: float
    vy: float
    life: int = 10

    @property
    def is_alive(self) -> bool:
        return self.life > 0

    def move(self) -> None:
        if self.is_alive:
            self.x += self.vx
            self.y += self.vy

    def take_damage(self, amount: int) -> None:
        self.life = max(0, self.life - amount)


@dataclass(kw_only=True)
class Tower:
    label: str
    x: float
    y: float
    attack_points: int
    attack_range: float

    def in_range(self, enemy: Enemy) -> bool:
        return math.hypot(self.x - enemy.x, self.y - enemy.y) <= self.attack_range

    def attack(self, enemies: list[Enemy]) -> None:
        for enemy in enemies:
            if enemy.is_alive and self.in_range(enemy):
                enemy.take_damage(self.attack_points)


class BasicTower(Tower):
    def __init__(self, label: str, x: float, y: float) -> None:
        super().__init__(label=label, x=x, y=y, attack_points=1, attack_range=2)


class AdvancedTower(Tower):
    def __init__(self, label: str, x: float, y: float) -> None:
        super().__init__(label=label, x=x, y=y, attack_points=2, attack_range=4)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_task1() -> None:
    print("=== Task 1 ===")
    line_a = Line(start=Point(x=-6, y=1), end=Point(x=2, y=4))
    line_b = Line(start=Point(x=-6, y=-1), end=Point(x=2, y=2))
    line_c = Line(start=Point(x=-1, y=6), end=Point(x=-4, y=-4))
    circle_a = Circle(center=Point(x=6, y=3), radius=2)
    circle_b = Circle(center=Point(x=8, y=1), radius=1)
    polygon_a = Polygon(
        vertices=[
            Point(x=2, y=0),
            Point(x=5, y=-1),
            Point(x=4, y=-4),
            Point(x=-1, y=-2),
        ]
    )

    print(f"Are Line A and Line B parallel? {line_a.is_parallel(line_b)}")
    print(f"Are Line C and Line A perpendicular? {line_c.is_perpendicular(line_a)}")
    print(f"Area of Circle A: {circle_a.area()}")
    print(f"Do Circle A and Circle B intersect? {circle_a.intersects(circle_b)}")
    print(f"Perimeter of Polygon A: {polygon_a.perimeter()}")


def run_task2() -> None:
    print("=== Task 2 ===")
    enemies = [
        Enemy(label="E1", x=-10, y=2, vx=2, vy=-1),
        Enemy(label="E2", x=-8, y=0, vx=3, vy=1),
        Enemy(label="E3", x=-9, y=-1, vx=3, vy=0),
    ]

    towers = [
        BasicTower(label="T1", x=-3, y=2),
        BasicTower(label="T2", x=-1, y=-2),
        BasicTower(label="T3", x=4, y=2),
        BasicTower(label="T4", x=7, y=0),
        AdvancedTower(label="A1", x=1, y=1),
        AdvancedTower(label="A2", x=4, y=-3),
    ]

    turns = 10
    for _ in range(turns):
        for enemy in enemies:
            enemy.move()
        for tower in towers:
            tower.attack(enemies)

    for enemy in enemies:
        print(f"{enemy.label}: position=({enemy.x}, {enemy.y}), life={enemy.life}")


def main() -> None:
    run_task1()
    print()
    run_task2()


if __name__ == "__main__":
    main()
