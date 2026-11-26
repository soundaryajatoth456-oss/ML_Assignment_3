"""Execute one notebook in a fresh kernel.

The kernel is the project virtual environment. The notebook's working
directory is the repository root, so both notebooks can be launched
without depending on the shell's current directory beyond this script.
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: execute_notebook.py <notebook.ipynb>")
    path = Path(sys.argv[1])
    if not path.is_absolute():
        path = ROOT / path
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=900,
        kernel_name="ml-a3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    print("executed", path)


if __name__ == "__main__":
    main()
