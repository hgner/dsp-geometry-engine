# Cloud analysis-only image (plan M9). No engine exe inside: the bridge's feature
# detection degrades extract_mesh_telemetry to a structured error while the analysis
# tools work on any PLY/palette/PNG synced under /data.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Dependency layer first: metadata only, so src/ edits do not bust the resolved
# third-party layer. --no-install-project defers building our own package.
# README.md is required here — pyproject's readme field is read at metadata build.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-dev --no-editable --no-install-project

# License layer: the image redistributes the GPL-3.0-or-later Blender workers alongside
# the Apache-2.0 rest of the tree, so it must carry its own grants. This must precede the
# project install — pyproject's license-files globs are resolved at metadata build time and
# hatchling silently emits zero License-File entries when they match nothing.
COPY LICENSE NOTICE ./
COPY LICENSES/ LICENSES/

# Project layer: copy sources, then install the package for real (--no-editable so
# the venv holds a genuine site-packages install, not a pointer back into /app/src).
COPY src/ src/
RUN uv sync --locked --no-dev --no-editable

# Non-root runtime user; /data is the telemetry mount (aws s3 sync / EFS / volume).
RUN useradd -m dsp && mkdir -p /data && chown -R dsp:dsp /app /data

# DSP_HOST is explicit: the server defaults to 127.0.0.1 so the MCP SDK's DNS-rebinding
# protection stays armed for local runs, but a container has to bind every interface to be
# reachable through the published port. Publish it behind auth — see the HTTP notes in the README.
ENV DSP_TRANSPORT=streamable-http \
    DSP_HOST=0.0.0.0 \
    MPLBACKEND=Agg \
    DSP_DATA_DIR=/data \
    PATH="/app/.venv/bin:$PATH"

VOLUME /data
EXPOSE 8000
USER dsp

CMD ["dsp-server"]
