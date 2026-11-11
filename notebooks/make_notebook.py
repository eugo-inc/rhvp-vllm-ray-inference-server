#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the sample notebook, the prompt set and the synthetic recording, or check that the committed ones are what
this script writes.

  python3 notebooks/make_notebook.py            # writes all three
  python3 notebooks/make_notebook.py --check    # exit 1 when a committed file is not what this script writes

The notebook's cells come from rhvp_bench/showing.py; this script only writes them as nbformat 4.5 JSON, every cell
with a stable id, every code cell with no outputs and `execution_count: null`, in the layout Jupyter itself saves
(sorted keys, one-space indent, source split into lines). The notebook, prompts.jsonl and recordings/synthetic.json
are therefore pure functions of the code in this folder. Never edit the .ipynb by hand. Standard library only.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rhvp_bench import prompts, replay, showing  # noqa: E402 (the folder is put on the path just above)

NOTEBOOK = HERE / "inference_timings.ipynb"
NBFORMAT, NBFORMAT_MINOR = 4, 5
METADATA = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def _cell(cell: showing.Cell) -> dict:
    entry: dict = {"cell_type": cell.kind, "id": cell.id, "metadata": {}, "source": cell.source.splitlines(True)}
    if cell.kind == showing.CODE:
        entry.update(execution_count=None, outputs=[])
    return entry


def notebook_text() -> str:
    """The whole notebook file."""
    notebook = {"cells": [_cell(cell) for cell in showing.cells()], "metadata": METADATA, "nbformat": NBFORMAT,
                "nbformat_minor": NBFORMAT_MINOR}
    return json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False, separators=(",", ": ")) + "\n"


def outputs(notebook: Path, prompt_file: Path, recording: Path) -> dict[Path, str]:
    """Every file this script writes, with its contents."""
    return {
        notebook: notebook_text(),
        prompt_file: prompts.text(prompts.make()),
        recording: replay.recording_text(replay.synthesize()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 when a committed file differs")
    parser.add_argument("--out", type=Path, default=NOTEBOOK, help="the notebook file")
    parser.add_argument("--prompts", type=Path, default=prompts.PATH, help="the prompt set")
    parser.add_argument("--recording", type=Path, default=replay.SYNTHETIC, help="the synthetic recording")
    args = parser.parse_args(argv)

    files = outputs(args.out, args.prompts, args.recording)
    if args.check:
        stale = [path for path, text in files.items()
                 if not path.is_file() or path.read_text(encoding="utf-8") != text]
        for path in stale:
            print(f"STALE: {path.name} is not what make_notebook.py writes; run python3 notebooks/make_notebook.py")
        if stale:
            return 1
        print(f"OK: {', '.join(path.name for path in files)} are what make_notebook.py writes")
        return 0

    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
