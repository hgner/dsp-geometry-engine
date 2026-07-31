# License map

This directory holds the license texts referenced by the repository. The map below is
directory-scoped on purpose — the set of Blender-side worker files changes between revisions, so an
enumeration by filename goes stale, while the directory boundary does not.

| Scope | License | Text |
| --- | --- | --- |
| Everything under `src/bodymesh_server/blender_scripts/**` | GPL-3.0-or-later | `GPL-3.0-or-later.txt` |
| Every other file in this repository | Apache-2.0 | root `LICENSE` (pointer: `Apache-2.0.txt`) |

The distribution as a whole is therefore `Apache-2.0 AND GPL-3.0-or-later`, which is the SPDX
expression recorded in `pyproject.toml`. Copyright 2026 hgner <hgner09@gmail.com>.

## Why the split holds

The `blender_scripts/` files are copyleft because they execute inside Blender and reach MPFB's GPL
service API. They are not linked into the MCP host: `bodymesh_server` launches them as separate OS
processes via `subprocess.Popen(argv, shell=False)` (`src/bodymesh_server/blender_bridge.py`) and the
only thing crossing that boundary is request/result JSON plus filesystem paths. No worker imports a
project module — at module scope they import stdlib plus Blender's own `bpy`/`mathutils`, and MPFB is
reached through `importlib.import_module` at runtime — so the Apache-2.0 side of the tree neither
imports nor is imported by the GPL side.

MPFB and Blender are installed separately by the operator. Neither is bundled in this repository or
in the Python distributions built from it; the wheel ships the worker sources only.
