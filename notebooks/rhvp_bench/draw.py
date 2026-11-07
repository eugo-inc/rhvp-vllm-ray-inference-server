# SPDX-License-Identifier: Apache-2.0
"""The notebook's HTML: the big side banners, the result cards, the neutral notice and the amber strips.

Every function returns markup as a string; `show()` displays it. Every style that uses a JupyterLab theme variable
is written twice, the plain colour first: a saved notebook's outputs open "untrusted", and JupyterLab then drops the
styles that use theme variables, so the plain colour still shows. Every visible string comes from words.py or from a
run's own figures, and is escaped.
"""

import html
import re

from . import words

#: WITHOUT EUGO in red, WITH EUGO in the Eugo violet.
COLOUR = {words.WITHOUT: "#F04242", words.WITH: "#A142FF"}
#: The same colours as a light background tint for the side banners.
TINT = {words.WITHOUT: "rgba(240,66,66,.09)", words.WITH: "rgba(161,66,255,.09)"}
AMBER = "#B26B00"
FONT = "var(--jp-ui-font-family, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif)"
INK = "var(--jp-ui-font-color0, #1b1b1f)"
SOFT = "var(--jp-ui-font-color1, #3b3b44)"
MUTED = "var(--jp-ui-font-color2, #6b6b75)"
BORDER = "var(--jp-border-color2, #dcdce2)"
WIDTH = "880px"

_STYLE = re.compile(r'style="([^"]*)"')
_VAR = re.compile(r"var\(--[\w-]+,\s*([^()]*)\)")


def doubled(markup: str) -> str:
    """`markup` with every style rule that uses a theme variable written twice: plain colour, then the variable."""
    def both(style: re.Match) -> str:
        rules = [rule.strip() for rule in style.group(1).split(";") if rule.strip()]
        out = []
        for rule in rules:
            if "var(" in rule:
                out.append(_VAR.sub(lambda v: v.group(1).strip(), rule))
            out.append(rule)
        return 'style="' + ";".join(out) + '"'
    return _STYLE.sub(both, markup)


def _text(value: object) -> str:
    return html.escape(str(value))


def side_banner(label: str) -> str:
    """The full-width banner that opens each side: WITHOUT EUGO in red or WITH EUGO in violet, and nothing else."""
    if label not in COLOUR:
        raise ValueError(f"a side banner is drawn only for {words.WITHOUT} or {words.WITH}")
    colour = COLOUR[label]
    return doubled(
        f'<div data-rhvp="side-banner" style="font-family:{FONT};box-sizing:border-box;width:100%;'
        f'margin:14px 0 10px;padding:18px 26px;border:3px solid {colour};border-left-width:14px;border-radius:10px;'
        f'background:{TINT[label]};color:{colour};font-size:32px;font-weight:800;letter-spacing:.14em;'
        f'line-height:1.2">{_text(label)}</div>'
    )


def strip(kind: str) -> str:
    """The one-line amber strip above a result: "INTERNAL" or "REPLAY"."""
    text = words.STRIP[kind]
    return doubled(
        f'<div data-rhvp="strip" style="font-family:{FONT};max-width:{WIDTH};margin:8px 0;padding:6px 12px;'
        f'font-size:12px;font-weight:600;letter-spacing:.04em;color:{AMBER};border:1px solid {AMBER};'
        f'border-radius:6px">{_text(text)}</div>'
    )


def result_card(label: str, value: str, unit: str, notes: list[str] | tuple[str, ...] = ()) -> str:
    """One WITHOUT EUGO result: its label, the big figure with its unit, then the notes. Only the WITHOUT EUGO side
    ever has a result card."""
    colour = COLOUR[words.WITHOUT]
    lines = "".join(f"<div>{_text(note)}</div>" for note in notes)
    return doubled(
        f'<div data-rhvp="result-card" style="font-family:{FONT};flex:1;min-width:240px;max-width:{WIDTH};'
        f'box-sizing:border-box;margin:8px 0;border:1px solid {BORDER};border-top:3px solid {colour};'
        f'border-radius:10px;padding:16px 22px 14px">'
        f'<div style="font-size:12px;font-weight:600;letter-spacing:.08em;color:{colour}">{_text(label)}</div>'
        f'<div style="font-size:46px;font-weight:700;line-height:1.15;margin-top:6px;color:{INK};'
        f'font-variant-numeric:tabular-nums">{_text(value)}'
        f'<span style="font-size:20px;font-weight:600;margin-left:8px;color:{MUTED}">{_text(unit)}</span></div>'
        f'<div style="font-size:13px;line-height:1.7;color:{SOFT};margin-top:6px">{lines}</div></div>'
    )


def card_row(cards: list[str], notes: list[str] | tuple[str, ...] = ()) -> str:
    """Several result cards side by side, then the notes they share."""
    lines = "".join(f"<div>{_text(note)}</div>" for note in notes)
    return doubled(
        f'<div style="font-family:{FONT};max-width:{WIDTH}">'
        f'<div style="display:flex;gap:16px;flex-wrap:wrap">{"".join(cards)}</div>'
        f'<div style="font-size:13px;line-height:1.7;color:{SOFT};margin:6px 0 4px">{lines}</div></div>'
    )


def notice(text: str, lines: list[str] | tuple[str, ...] = ()) -> str:
    """A neutral grey box: the WITH EUGO side's one line, or a plain message in place of a result."""
    extra = "".join(f'<div style="font-size:13px;line-height:1.6;color:{MUTED}">{_text(n)}</div>' for n in lines)
    return doubled(
        f'<div data-rhvp="notice" style="font-family:{FONT};max-width:{WIDTH};box-sizing:border-box;margin:8px 0;'
        f'padding:12px 16px;border:1px solid {BORDER};border-left:3px solid {MUTED};border-radius:8px">'
        f'<div style="font-size:14px;font-weight:600;color:{INK}">{_text(text)}</div>{extra}</div>'
    )


def quiet_line(text: str, ok: bool = True) -> str:
    """One short line of the readiness cell: a tick (done) or a dot (still waiting), then the phrase."""
    mark, colour = ("✓", INK) if ok else ("…", AMBER)
    return doubled(
        f'<div style="font-family:{FONT};font-size:13px;color:{SOFT};margin:4px 0">'
        f'<span style="color:{colour};font-weight:700;display:inline-block;width:1.4em">{mark}</span>'
        f'{_text(text)}</div>'
    )


def show(*parts: str):
    """Display the parts as one HTML output; returns its display handle, for `update`."""
    from IPython.display import HTML, display

    return display(HTML("".join(parts)), display_id=True)


def update(handle, *parts: str) -> None:
    """Replace what `show` displayed under `handle` with the parts."""
    from IPython.display import HTML

    if handle is not None:
        handle.update(HTML("".join(parts)))
