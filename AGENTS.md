# DSP-Geometry-Engine (proje10)

@llms.txt

Codex-only notes:
- Use `uv run` for everything (`uv run pytest`, `uv run ruff check .`, `uv run dsp-server`) — never bare `python`.
- Never start the MCP server manually inside a session — `.codex/config.toml` auto-registers it for
  trusted Codex projects (`.mcp.json` does the same for Claude Code); a manual stdio instance just hangs.
- The C++ engine lives in `C:\Users\hgner\hakantest\proje7-engine` (separate repo/worktree) — always plan before touching it.
- Everything under `data/` (dumps, cache, plots, logs) is disposable telemetry — safe to delete, regenerated on demand.
