"""Sample Python module for AST parser smoke tests."""

from __future__ import annotations

import os
from typing import Iterable


class Greeter:
    """Simple greeter class."""

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        return format_greeting(self.name)


def format_greeting(name: str) -> str:
    """Return a greeting string."""
    return f"Hello, {name}!"


def main(names: Iterable[str]) -> None:
    for name in names:
        greeter = Greeter(name)
        print(greeter.greet())
        print(os.getcwd())


if __name__ == "__main__":
    main(["world"])
