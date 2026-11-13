# SPDX-License-Identifier: Apache-2.0
"""The comparison notebook's cells, in order: what make_notebook.py writes into compare_images.ipynb.

A title cell, one code cell that lists the runs, then five numbered sections, each a heading cell and one short code
cell. Every sentence comes from words.py. Standard library only.
"""

from rhvp_bench.showing import CODE, MARKDOWN, Cell

from . import words

__all__ = ["CODE", "MARKDOWN", "Cell", "cells"]  # CODE and MARKDOWN are part of this module's API

IMPORT = "from rhvp_compare import show"
RUNS_CALL = "show.runs()"
SECTION_CALL = {1: "show.verdicts()", 2: "show.startup()", 3: "show.energy()", 4: "show.scale_out()",
                5: "show.provenance()"}


def cells() -> list[Cell]:
    """Every cell of the comparison notebook, in order, each with a stable id."""
    out = [
        Cell("title", MARKDOWN, "\n\n".join((f"# {words.TITLE}", words.INTRO, words.INTRO_HOW))),
        Cell("runs", CODE, "\n".join((IMPORT, "", RUNS_CALL))),
    ]
    for number in sorted(words.SECTION_TITLE):
        heading = words.SECTION_HEADING.format(number=number, title=words.SECTION_TITLE[number])
        out.append(Cell(f"s{number}", MARKDOWN, "\n\n".join((heading, words.SECTION_TEXT[number]))))
        out.append(Cell(f"s{number}-show", CODE, SECTION_CALL[number]))
    return out
