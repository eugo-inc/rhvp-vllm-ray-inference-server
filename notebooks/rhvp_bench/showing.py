# SPDX-License-Identifier: Apache-2.0
"""The notebook's cells, in order: what make_notebook.py writes into inference_timings.ipynb.

The layout: a title cell; one unnumbered cell that imports the helpers and waits until the server is ready; then four
numbered sections. Each section is a heading cell (what is measured), then one short code cell that measures the
server and shows the result.

Every sentence comes from words.py. Imports nothing that draws or measures, so the generator needs only the standard
library.
"""

from typing import NamedTuple

from . import words

MARKDOWN, CODE = "markdown", "code"
IMPORT = "from rhvp_bench import run"
READY_CALL = "run.ready()"
#: Each section's code cell.
SECTION_CALL = {1: "run.startup()", 2: "run.first_token()", 3: "run.decode()", 4: "run.throughput()"}


class Cell(NamedTuple):
    id: str
    kind: str
    source: str


def _markdown(cell_id: str, *paragraphs: str) -> Cell:
    return Cell(cell_id, MARKDOWN, "\n\n".join(paragraphs))


def _code(cell_id: str, *lines: str) -> Cell:
    return Cell(cell_id, CODE, "\n".join(lines))


def cells() -> list[Cell]:
    """Every cell of the notebook, in order, each with a stable id."""
    out = [
        _markdown("title", f"# {words.TITLE}", words.INTRO, words.INTRO_HOW),
        _code("ready", IMPORT, "", READY_CALL),
    ]
    for number in sorted(words.SECTION_TITLE):
        out.append(_markdown(f"s{number}", words.SECTION_HEADING.format(number=number,
                                                                        title=words.SECTION_TITLE[number]),
                             words.SECTION_TEXT[number]))
        out.append(_code(f"s{number}-run", SECTION_CALL[number]))
    return out
