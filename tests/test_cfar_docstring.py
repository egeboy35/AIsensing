"""The CFAR threshold docstring carries LaTeX, so it has to be a raw string.

`compute_thresholds` inside `cfar_2d_advanced` documents the three CFAR rules
in LaTeX. In an ordinary string literal Python resolves the escapes it knows
before anything else sees them, and two of the sequences in that block are
ones it knows:

    \\frac  ->  U+000C form feed, then "rac"
    \\text  ->  U+0009 tab,       then "ext"

so the formula a reader gets from `help()` is not the formula that was written.
The rest (`\\[`, `\\sum`, `\\max`, `\\min`, `\\Delta`) are invalid escapes: kept
verbatim today, a SyntaxWarning now, and slated to become a SyntaxError.

Reads the source rather than importing it, so no torch and no scipy are needed.

    pytest tests/test_cfar_docstring.py
"""
import ast
import io
import warnings
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "AIRadar" / "AIRadarLib" / "radar_det.py"
FORM_FEED = "\x0c"
TAB = "\t"

LATEX_COMMANDS = ["frac", "text", "sum", "max", "min", "Delta"]


def _docstring():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(io.open(SOURCE, encoding="utf-8", errors="replace").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "compute_thresholds":
            doc = ast.get_docstring(node, clean=False)
            if doc:
                return doc
    pytest.fail("compute_thresholds has no docstring any more")


def test_the_source_file_is_where_it_is_expected():
    assert SOURCE.is_file()


def test_the_docstring_still_documents_the_three_rules():
    doc = _docstring()
    for rule in ("CA-CFAR", "GO-CFAR", "SO-CFAR"):
        assert rule in doc


@pytest.mark.parametrize("command", LATEX_COMMANDS)
def test_every_latex_command_survives_parsing(command):
    assert "\\" + command in _docstring()


def test_no_control_character_was_produced_by_an_escape():
    doc = _docstring()
    assert FORM_FEED not in doc, "\\frac became a form feed"
    assert TAB not in doc, "\\text became a tab"


def test_the_display_math_delimiters_are_intact():
    doc = _docstring()
    assert doc.count("\\[") == 3
    assert doc.count("\\]") == 3


def test_the_file_emits_no_syntax_warning():
    """Invalid escapes are a warning today and a SyntaxError later."""
    src = io.open(SOURCE, encoding="utf-8", errors="replace").read()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile(src, str(SOURCE), "exec")
    offenders = [f"line {w.lineno}: {w.message}" for w in caught
                 if issubclass(w.category, SyntaxWarning)]
    assert offenders == []
