"""Neo4j schema: constraints and indexes for the call-graph layer.

Idempotent Cypher DDL so ``initialize_schema`` is safe to re-run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import Driver

# Composite uniqueness for callable / type definitions (file + name + kind).
CONSTRAINT_STATEMENTS: list[str] = [
    """
    CREATE CONSTRAINT function_identity IF NOT EXISTS
    FOR (n:Function)
    REQUIRE (n.file_path, n.name, n.node_type) IS UNIQUE
    """,
    """
    CREATE CONSTRAINT class_identity IF NOT EXISTS
    FOR (n:Class)
    REQUIRE (n.file_path, n.name, n.node_type) IS UNIQUE
    """,
    """
    CREATE CONSTRAINT file_path_unique IF NOT EXISTS
    FOR (n:File)
    REQUIRE n.path IS UNIQUE
    """,
    """
    CREATE CONSTRAINT module_name_unique IF NOT EXISTS
    FOR (n:Module)
    REQUIRE n.name IS UNIQUE
    """,
]

# Secondary indexes for lookup by file path.
INDEX_STATEMENTS: list[str] = [
    """
    CREATE INDEX function_file_path IF NOT EXISTS
    FOR (n:Function)
    ON (n.file_path)
    """,
    """
    CREATE INDEX class_file_path IF NOT EXISTS
    FOR (n:Class)
    ON (n.file_path)
    """,
    """
    CREATE INDEX function_name IF NOT EXISTS
    FOR (n:Function)
    ON (n.name)
    """,
]


def initialize_schema(driver: Driver) -> None:
    """Create uniqueness constraints and indexes (idempotent).

    Args:
        driver: An open Neo4j driver instance.
    """
    with driver.session() as session:
        for statement in CONSTRAINT_STATEMENTS:
            session.run(statement.strip())
        for statement in INDEX_STATEMENTS:
            session.run(statement.strip())
