import os
from pyan import create_callgraph

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
py_files = [os.path.join(PROJECT_ROOT, f) for f in os.listdir(PROJECT_ROOT) if f.endswith(".py")]

data = create_callgraph(
    filenames=py_files,  # absolute paths
    root=None,           # avoid pyan trying to compute package root
    format="dot",
    draw_uses=True,
    draw_defines=False
)

with open("call_graph.dot", "w", encoding="utf-8") as f:
    f.write(data)
