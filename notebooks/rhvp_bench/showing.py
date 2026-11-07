# SPDX-License-Identifier: Apache-2.0
"""The notebook's cells, in order: what make_notebook.py writes into stock_vs_eugo_inference.ipynb.

The layout: a title cell; one unnumbered cell that imports the helpers and waits until the server is ready; then four
numbered sections. Each section is a heading cell (what is measured), then the WITHOUT EUGO side (a markdown banner
heading and one short code cell that measures and shows the result); sections 1 and 2 then have the WITH EUGO side (a
markdown banner heading and one code cell that shows the WITH EUGO banner and the neutral line, and nothing else).
Sections 3 and 4 have no WITH EUGO side.

Every sentence comes from words.py. Imports nothing that draws or measures, so the generator needs only the standard
library.
"""

from typing import NamedTuple

from . import words

MARKDOWN, CODE = "markdown", "code"
#: The sections that have a WITH EUGO side.
WITH_EUGO_SECTIONS = (1, 2)
IMPORT = "from rhvp_bench import run"
READY_CALL = "run.ready()"
#: Each section's WITHOUT EUGO cell.
SECTION_CALL = {1: "run.startup()", 2: "run.first_token()", 3: "run.decode()", 4: "run.throughput()"}
WITH_EUGO_CALL = "run.with_eugo()"


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
        _markdown("title", f"# {words.TITLE}", words.TITLE_BANNER, words.INTRO, words.INTRO_HOW),
        _code("ready", IMPORT, "", READY_CALL),
    ]
    for number in sorted(words.SECTION_TITLE):
        out.append(_markdown(f"s{number}", words.SECTION_HEADING.format(number=number,
                                                                        title=words.SECTION_TITLE[number]),
                             words.SECTION_TEXT[number]))
        out.append(_markdown(f"s{number}-without-heading", words.SIDE_HEADING[words.WITHOUT]))
        out.append(_code(f"s{number}-without", SECTION_CALL[number]))
        if number in WITH_EUGO_SECTIONS:
            out.append(_markdown(f"s{number}-with-heading", words.SIDE_HEADING[words.WITH]))
            out.append(_code(f"s{number}-with", WITH_EUGO_CALL))
    return out
