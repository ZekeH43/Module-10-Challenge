"""Execute the notebook in a clean kernel from its resource directory.

Run after installing requirements.txt; errors cause nonzero exit status.
No outputs are written into the tracked notebook.
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Starter_Code" / "crypto_investments.ipynb"


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    runner = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    executed = runner.execute()
    for cell in executed.cells:
        if cell.cell_type == "code":
            for output in cell.get("outputs", []):
                if output.output_type == "error":
                    raise AssertionError(
                        f"Notebook error: {output.ename}: {output.evalue}"
                    )
    print(f"Executed notebook successfully: {NOTEBOOK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
