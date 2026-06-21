def setup_graph(session, vocab: dict, color_batch_size: int) -> None:
    """Create the vector index and MERGE all Color/ColorFamily nodes."""
    session.run("DROP INDEX color_lab_index IF EXISTS")
    session.run("""
        CREATE VECTOR INDEX color_lab_index
        FOR (c:Color) ON c.lab_vector
        OPTIONS {indexConfig: {
            `vector.dimensions`: 3,
            `vector.similarity_function`: 'euclidean'
        }}
    """)

    colors_data = [
        {
            "name":       vocab["names"][i],
            "hex":        vocab["hexes"][i],
            "r":          int(vocab["rgbs"][i][0]),
            "g":          int(vocab["rgbs"][i][1]),
            "b":          int(vocab["rgbs"][i][2]),
            "l":          float(vocab["labs"][i][0]),
            "a_lab":      float(vocab["labs"][i][1]),
            "b_lab":      float(vocab["labs"][i][2]),
            "lab_vector": vocab["labs"][i].tolist(),
            "family":     vocab["families"][i],
        }
        for i in range(len(vocab["names"]))
    ]

    for i in range(0, len(colors_data), color_batch_size):
        session.run("""
            UNWIND $colors AS c
            MERGE (color:Color {name: c.name})
            SET color += {hex: c.hex, r: c.r, g: c.g, b: c.b,
                          l: c.l, a_lab: c.a_lab, b_lab: c.b_lab,
                          lab_vector: c.lab_vector, family: c.family}
        """, colors=colors_data[i:i + color_batch_size])

    session.run("""
        MATCH (c:Color)
        MERGE (f:ColorFamily {name: c.family})
        MERGE (c)-[:IN_FAMILY]->(f)
    """)


def write_paintings(session, results: list[dict], batch_size: int) -> None:
    """MERGE Painting nodes and their HAS_COLOR relationships."""
    for i in range(0, len(results), batch_size):
        session.run("""
            UNWIND $paintings AS p
            MERGE (painting:Painting {path: p.path})
            SET painting.name = p.name,
                painting.width = p.width,
                painting.height = p.height
            WITH painting, p
            UNWIND p.colors AS c
            MATCH (color:Color {name: c.name})
            MERGE (painting)-[h:HAS_COLOR]->(color)
            SET h.percentage = c.percentage, h.rank = c.rank
        """, paintings=results[i:i + batch_size])
