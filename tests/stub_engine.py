"""CI stand-in for the PATCHED layered_field_dump_cli (M5 flag surface included).

Launched by the bridge as ``[sys.executable, stub_engine.py]`` (set DSP_ENGINE_CLI
to this file). It honors the real CLI's argument grammar, exit codes, and stderr
protocol verbatim in shape, and writes a 64x16 synthetic forearm cylinder through
the same ``write_engine_ply`` the parser is tested against — so bridge tests
exercise discovery, feature detection, naming, teeing, and parsing end to end
without a C++ build. Hidden ``--sleep <s>`` exists only for timeout tests.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# This script runs as a child process with an arbitrary cwd — make 'synth' (tests/)
# and 'dsp_server' (src/) importable regardless of how/where it was spawned.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

N_RINGS = 64
N_AROUND = 16
CORRUGATION_AMP = 0.002  # meters, injected only when --pose is given
CORRUGATION_FREQ_CPM = 60.0  # cycles/meter, safely under the axial Nyquist

# --help text MUST advertise the M5 flags: the bridge feature-detects
# --character/--time/--palette-out/--weights from this exact text.
USAGE = """\
usage: layered_field_dump_cli --sex male|female --out <path> [--pose <clipId>] [--list-sources]
                              [--character <baked.json>] [--time <sec>] [--palette-out <json>] [--weights]
                              [--bust-cm 90] [--waist-cm 62] [--hips-cm 94] [--height-cm 170]
  (no flag) rest field PLY; (--pose) posed mesh by dominant joint; (--list-sources) all sources JSON
  (--character) import a baked character; (--time) sampleTime override
  (--palette-out) skinning palette sidecar; (--weights) append j0..j3/w0..w3 vertex properties
"""

# A tiny 3-source --list-sources payload in the real dumper's row shape
# (i/kind/a/b/r0/r1/center/radii/normal/blendK/sub/bone), kept as literal JSON text.
TINY_SOURCE_COUNT = 3
TINY_SOURCES_JSON = """\
[
 {"i":0,"kind":"capsule","a":[0.05,1.25,0.02],"b":[0.09,1.21,0.04],"r0":0.05,"r1":0.045,
  "center":[0,0,0],"radii":[0,0,0],"normal":[0,1,0],"blendK":8.0,"sub":0,"bone":4},
 {"i":1,"kind":"capsule","a":[0.1,1.2,0.05],"b":[0.175,1.425,0.1],"r0":0.04,"r1":0.03,
  "center":[0,0,0],"radii":[0,0,0],"normal":[0,1,0],"blendK":8.0,"sub":0,"bone":5},
 {"i":2,"kind":"ellipsoid","a":[0,0,0],"b":[0,0,0],"r0":0.0,"r1":0.0,
  "center":[0.18,1.43,0.11],"radii":[0.03,0.02,0.02],"normal":[0,1,0],"blendK":6.0,"sub":0,"bone":6}
]
"""


def _err(message: str) -> None:
    print(f"layered_field_dump_cli: {message}", file=sys.stderr)


def _write_palette(palette_out: str, pose: str, time_s: float | None, character: str, sex: str) -> None:
    from synth import STUB_BONE_MAP

    # 16 floats, column-major identity (identity is major-agnostic by symmetry).
    identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    bones = [
        {
            "index": index,
            "name": name,
            "parent": index - 1,  # simple chain: -1, 0, 1, ...
            "skinning": identity,
            "world": identity,
            "inverseBind": identity,
        }
        for index, name in sorted(STUB_BONE_MAP.items())
    ]
    payload = {
        "clipId": pose or "rest",
        "sampleTimeSeconds": time_s if time_s is not None else 0.0,
        "character": character or None,
        "sex": sex,
        "matrixLayout": "column-major",
        "importScale": 1.0,
        "boneCount": len(bones),
        "bones": bones,
    }
    Path(palette_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str]) -> int:  # deliberately mirrors the C++ arg loop
    sex = "female"  # the real CLI's default
    out_path = ""
    pose = ""
    character = ""
    time_s: float | None = None
    palette_out = ""
    weights = False
    list_sources = False
    sleep_s = 0.0
    bust = waist = hips = height = 0.0

    def _float(raw: str) -> float:
        try:
            return float(raw)
        except ValueError:
            return 0.0  # strtof semantics

    i = 0
    while i < len(argv):
        arg = argv[i]
        has_value = i + 1 < len(argv)
        if arg in ("--help", "-h"):
            print(USAGE, end="")
            return 0
        elif arg == "--list-sources":
            list_sources = True
        elif arg == "--weights":
            weights = True
        elif arg == "--sex" and has_value:
            i += 1
            sex = argv[i]
            if sex not in ("male", "female"):
                _err("--sex must be male|female")
                return 2
        elif arg == "--out" and has_value:
            i += 1
            out_path = argv[i]
        elif arg == "--pose" and has_value:
            i += 1
            pose = argv[i]
        elif arg == "--character" and has_value:
            i += 1
            character = argv[i]
        elif arg == "--time" and has_value:
            i += 1
            time_s = _float(argv[i])
        elif arg == "--palette-out" and has_value:
            i += 1
            palette_out = argv[i]
        elif arg == "--sleep" and has_value:  # hidden: timeout tests only
            i += 1
            sleep_s = _float(argv[i])
        elif arg == "--bust-cm" and has_value:
            i += 1
            bust = _float(argv[i])
        elif arg == "--waist-cm" and has_value:
            i += 1
            waist = _float(argv[i])
        elif arg == "--hips-cm" and has_value:
            i += 1
            hips = _float(argv[i])
        elif arg == "--height-cm" and has_value:
            i += 1
            height = _float(argv[i])
        else:
            _err(f"unknown arg '{arg}'")
            return 2
        i += 1

    if sleep_s > 0.0:
        time.sleep(sleep_s)

    if not out_path:
        _err("--out is required")
        return 2

    # Mirror of the patched CLI's guard (main.cpp): posed-only levers without --pose -> exit 2.
    if not pose and (character or time_s is not None or palette_out or weights):
        _err(
            "--character/--time/--palette-out/--weights require --pose "
            "(the posed PLY already carries rest positions)"
        )
        return 2

    _err(f"body-measurements bust={bust:g} waist={waist:g} hips={hips:g} height={height:g}")

    if list_sources:
        Path(out_path).write_text(TINY_SOURCES_JSON, encoding="utf-8")
        print(
            f"layered_field_dump_cli-sources-ok n={TINY_SOURCE_COUNT} out={out_path}",
            file=sys.stderr,
        )
        return 0

    if character:
        rejected = not Path(character).is_file()
        if not rejected:
            try:
                json.loads(Path(character).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                rejected = True
        if rejected:
            _err(f"--character rejected (missing file or invalid baked-character json): {character}")
            return 1

    # Heavy imports are deferred so --sleep/arg-error paths stay numpy-free and fast.
    from dsp_server.engine.ply import write_engine_ply
    from synth import STUB_BONE_MAP, make_cylinder, stub_bone_map_stderr

    dump = make_cylinder(
        n_rings=N_RINGS,
        n_around=N_AROUND,
        corrugation_amp=CORRUGATION_AMP if pose else 0.0,
        corrugation_freq_cpm=CORRUGATION_FREQ_CPM,
        with_weights=weights,
    )
    write_engine_ply(Path(out_path), dump)

    _err("deformers applied (collisionPush=0.0012 bakedCollisionPush=0)")
    if character:
        print(
            f"import-ok verts={dump.vertex_count} bones={len(STUB_BONE_MAP)} clips=3 packClips=42",
            file=sys.stderr,
        )
    if pose:
        sample_time = time_s if time_s is not None else 0.0
        print(
            f"layered_field_dump_cli-posed-ok clip={pose} sampleTime={sample_time:g} "
            f"verts={dump.vertex_count} out={out_path}",
            file=sys.stderr,
        )
    else:
        print(f"layered_field_dump_cli-ok sex={sex} out={out_path}", file=sys.stderr)
    print(stub_bone_map_stderr(), file=sys.stderr)

    if palette_out:
        _write_palette(palette_out, pose, time_s, character, sex)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
