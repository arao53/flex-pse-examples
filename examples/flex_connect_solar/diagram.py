"""The Flex Connect + solar flowsheet as a diagram: units on a grid, lines between ports.

Rendered by flex_pse_examples.diagrams -- live in notebook.py, and into
public/flex_connect_solar/flowsheet.svg for the WebAssembly page by
tools/render_diagrams.py. Keep it in step with the blocks model.py wires.
"""

NODES = [
    {"id": "water", "kind": "terminal", "c": 0, "r": 0, "label": "Water in"},
    {"id": "process", "icon": "filter", "c": 1, "r": 0, "label": "Process (5 MW)"},
    {"id": "product", "kind": "terminal", "c": 2, "r": 0, "label": "Product (95% avg)"},
    {"id": "bus", "kind": "junction", "c": 1, "r": 1},
    {"id": "solar", "icon": "solar-panel", "c": 0, "r": 1, "label": "Solar (1 MW)"},
    {"id": "grid", "icon": "grid", "c": 2, "r": 1, "label": "PG&E Flex Connect"},
]

EDGES = [
    ("water.e", "process.w", "fluid"),
    ("process.e", "product.w", "fluid"),
    ("solar.e", "bus.w", "electricity"),
    ("grid.w", "bus.e", "electricity"),
    ("bus.n", "process.s", "electricity"),
]
