class Estado:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))


class Acao:
    def __init__(self, dx: int, dy: int):
        self.dx = dx
        self.dy = dy

    def __eq__(self, other) -> bool:
        return self.dx == other.dx and self.dy == other.dy

    def __hash__(self) -> int:
        return hash((self.dx, self.dy))
