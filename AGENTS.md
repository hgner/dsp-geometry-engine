# DSP-Geometry-Engine (proje10)

@llms.txt

Codex-only notes:
- Use `uv run` for everything (`uv run pytest`, `uv run ruff check .`) — never bare `python`.
- Never start either MCP entrypoint (`uv run dsp-server` or `uv run bodymesh-server`) manually inside
  a session. Copy `.codex/config.toml.example` to `.codex/config.toml` (untracked) once; it then
  auto-registers the independent `dsp-geometry-engine` and `blender-body-mesh` processes for trusted
  Codex projects (`.mcp.json.example` -> `.mcp.json` does the same for Claude Code). An extra stdio
  instance waits for JSON-RPC and appears hung.
- The C++ engine is a separate repo/worktree, not part of this one. Its checkout is located via
  `DSP_ENGINE_ROOT` (`<engine-checkout>`, e.g. `C:/path/to/proje7-engine`), or pinned to one exe with
  `DSP_ENGINE_CLI` — always plan before touching it. Exactly one of the 49 DSP tools
  (`extract_mesh_telemetry`) needs it; `DSP_ENGINE_CLI=tests/stub_engine.py` covers that one without
  any C++.
- Everything under `data/` (dumps, cache, plots, logs) is disposable telemetry — safe to delete, regenerated on demand.
