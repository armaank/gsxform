"""Render the marimo example notebooks into docs pages at build time.

A nav entry `examples/<name>.md` is generated from `examples/<name>.py`: the
notebook runs in the build's own environment, and the page gets its markdown,
its code and its figures as static content. mkdocs-marimo cannot do this, since
it runs notebooks in the browser with Pyodide, where torch does not exist.

Printed output is not carried over, only each cell's displayed value.

Rendered pages are cached in `.cache/marimo-examples/`, keyed by the notebook
and the gsxform sources, so `mkdocs serve` does not re-run training on every
edit.
"""

import ast
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from textwrap import dedent
from typing import Any

import marimo
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files

log = logging.getLogger("mkdocs.hooks.marimo_examples")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "marimo-examples"


def _nav_pages(items: Any) -> list[str]:
    """Every page path in the nav, flattened."""
    if isinstance(items, str):
        return [items]
    if isinstance(items, list):
        return [page for item in items for page in _nav_pages(item)]
    if isinstance(items, dict):
        return [page for value in items.values() for page in _nav_pages(value)]
    return []


def _markdown_source(code: str) -> str | None:
    """Return the text of a `mo.md("...")` cell, or None for any other cell."""
    try:
        (statement,) = ast.parse(code).body
    except (SyntaxError, ValueError):
        return None
    call = getattr(statement, "value", None)
    if (
        isinstance(call, ast.Call)
        and ast.unparse(call.func) == "mo.md"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    ):
        return dedent(call.args[0].value).strip()
    return None


def _output_html(mimetype: str, data: Any) -> str:
    """Return static HTML for one cell output."""
    if mimetype == "application/vnd.marimo+mimebundle":
        bundle = json.loads(data) if isinstance(data, str) else data
        for inner in ("image/png", "text/html", "text/plain"):
            if inner in bundle:
                return _output_html(inner, bundle[inner])
        return ""
    if mimetype in ("image/png", "image/svg+xml", "image/jpeg"):
        return f'<img src="{data}" alt="">'
    if mimetype in ("text/html", "text/markdown"):
        return f'<div class="marimo-output">{data}</div>'
    if mimetype == "text/plain" and data:
        return f"<pre>{data}</pre>"
    return ""


def _render(notebook: Path) -> str:
    """Run a marimo notebook and return its page as markdown."""
    generator = marimo.MarimoIslandGenerator.from_file(str(notebook))
    asyncio.run(generator.build())

    parts = []
    for stub in generator.stubs:
        text = _markdown_source(stub.code)
        if text is not None:
            parts.append(text)
            continue
        parts.append(f"```python\n{stub.code.strip()}\n```")
        output = stub.output
        if output is not None:
            html = _output_html(output.mimetype, output.data)
            if html:
                parts.append(html)
    return "\n\n".join(parts) + "\n"


def _cache_key(notebook: Path) -> str:
    """Hash of the notebook and every gsxform source file."""
    digest = hashlib.sha256(notebook.read_bytes())
    for source in sorted((ROOT / "gsxform").glob("*.py")):
        digest.update(source.read_bytes())
    return digest.hexdigest()[:16]


def _page(notebook: Path) -> str:
    """Return the rendered page, from the cache when the inputs are unchanged."""
    cached = CACHE / f"{notebook.stem}-{_cache_key(notebook)}.md"
    if cached.exists():
        return cached.read_text()
    log.info(f"Running {notebook.relative_to(ROOT)}")
    page = _render(notebook)
    CACHE.mkdir(parents=True, exist_ok=True)
    for stale in CACHE.glob(f"{notebook.stem}-*.md"):
        stale.unlink()
    cached.write_text(page)
    return page


def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Add a generated page for every example notebook in the nav."""
    for path in _nav_pages(config.nav):
        if not (path.startswith("examples/") and path.endswith(".md")):
            continue
        notebook = ROOT / "examples" / f"{Path(path).stem}.py"
        if notebook.exists():
            files.append(File.generated(config, path, content=_page(notebook)))
    return files
