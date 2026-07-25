"""Delete all Painting nodes and their HAS_COLOR edges.

Keeps the Color / ColorFamily nodes and the vector index. Run this before a
re-ingest: pipeline.py MERGEs paintings by path and colours by name, so running
it repeatedly on the same images accumulates duplicate HAS_COLOR edges (each run
samples pixels randomly and can pick slightly different nearest colours). A clean
slate keeps each painting at ~n_colors edges summing to ~1.0.
"""
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

_ENV = os.path.join(os.path.dirname(__file__), "..", ".env")


def main() -> None:
    load_dotenv(_ENV)
    pwd = os.environ.get("NEO4J_PASSWORD", "")
    auth = (os.environ.get("NEO4J_USER", "neo4j"), pwd) if pwd else None
    driver = GraphDatabase.driver(os.environ["NEO4J_URI"], auth=auth)
    with driver.session() as session:
        before = session.run("MATCH (p:Painting) RETURN count(p) AS c").single()["c"]
        session.run("MATCH (p:Painting) DETACH DELETE p")
        after = session.run("MATCH (p:Painting) RETURN count(p) AS c").single()["c"]
    driver.close()
    print(f"Reset graph: {before} → {after} Painting nodes (Color/ColorFamily + index kept)")


if __name__ == "__main__":
    main()
