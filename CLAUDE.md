# DSP-Geometry-Engine (proje10)

@llms.txt

Claude-Code-only notes:
- Use `uv run` for everything (`uv run pytest`, `uv run ruff check .`, `uv run dsp-server`) — never bare `python`.
- Never start the MCP server manually inside a session — `.mcp.json` auto-registers it; a manual stdio instance just hangs.
- The C++ engine lives in `C:\Users\hgner\hakantest\proje7-engine` (separate repo/worktree) — always plan before touching it.
- Everything under `data/` (dumps, cache, plots, logs) is disposable telemetry — safe to delete, regenerated on demand.
