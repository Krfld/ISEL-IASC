class Acao:
    def __init__(self, dx: int, dy: int):
        self.dx = dx
        self.dy = dy

    def __eq__(self, other) -> bool:
        return self.dx == other.dx and self.dy == other.dy

    def __hash__(self) -> int:
        return hash((self.dx, self.dy))
