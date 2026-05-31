# Tech Conventions

## Python

- Target Python **≥ 3.11** (`pyproject.toml` enforces).
- Use built-in `asyncio` for concurrency. No `trio`, no `anyio`.
- `pydantic` v2 for all data contracts. `dataclasses` for purely
  internal value types that don't cross the wire.
- `loguru` for logging. `print()` is reserved for CLI user output via
  `rich.console`.

## Imports

- Order: stdlib → third-party → first-party (`dr_agent.*`).
- No relative imports across modules (`from dr_agent.x.y import …`).
- Lazy-import heavy optionals (`sentence_transformers`, `streamlit`,
  `fitz`) inside the function that needs them.

## Errors

- Use the layered exception hierarchy in `dr_agent.llm.errors`
  (`LLMRateLimited` / `LLMTransient` / `LLMPermanent` / `LLMUnavailable`).
- Agents never raise into the DAG; wrap with `ResultEnvelope.failure`.
- CLI entry points let exceptions bubble to surface a clear error message
  (Typer prints the message; the global `pretty_exceptions_enable=False`
  keeps tracebacks compact).

## Async

- Each LLM call must occupy a `MimoPool` slot. Don't bypass.
- CPU-bound work (textrank, sklearn) goes through `asyncio.to_thread`
  with a `lambda` if kwargs are needed.
- Always close async clients via `async with` or explicit `aclose()`.

## Tests

- `pytest` + `pytest-asyncio` (mode = "auto").
- Mock external HTTP via `monkeypatch` on `httpx.AsyncClient.post` (or
  `respx` for more elaborate cases).
- One assertion focus per test. Combine setup with `_make_*` helpers.

## Windows + Anaconda gotchas

- Set `KMP_DUPLICATE_LIB_OK=TRUE` before any torch import. Both
  `cli.py`, `experiments/*.py`, and `embedder.py` already do this; new
  entry points must too.
- jieba's `dict.txt` can be missing in some pip installs;
  `pip install --force-reinstall --no-deps jieba` fixes it.

## Security

- `.env` is gitignored. All keys are masked via `mask_key()` in logs.
- Fetcher applies SSRF protection (no localhost / RFC1918 / link-local).
- LLM-returned text is truncated to ≤ 8 KB before being persisted to
  SQLite.
