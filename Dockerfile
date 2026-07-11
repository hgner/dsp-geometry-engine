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

# Project layer: copy sources, then install the package for real (--no-editable so
# the venv holds a genuine site-packages install, not a pointer back into /app/src).
COPY src/ src/
RUN uv sync --locked --no-dev --no-editable

# Non-root runtime user; /data is the telemetry mount (aws s3 sync / EFS / volume).
RUN useradd -m dsp && mkdir -p /data && chown -R dsp:dsp /app /data

ENV DSP_TRANSPORT=streamable-http \
    MPLBACKEND=Agg \
    DSP_DATA_DIR=/data \
    PATH="/app/.venv/bin:$PATH"

VOLUME /data
EXPOSE 8000
USER dsp

CMD ["dsp-server"]
