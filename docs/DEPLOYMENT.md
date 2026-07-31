# Deployment

The analysis server is pure Python + scipy. The container image ships no engine binary, which costs
exactly one of the 49 tools — see "Cloud analysis-only mode" below. Transport in containers is
`streamable-http` (set by the Dockerfile) and requires `DSP_AUTH_TOKEN`; tool code is identical to
the local stdio deployment.

> **Scope: this guide deploys only `dsp-geometry-engine` (`dsp-server`).** The Dockerfile,
> Compose service, port 8000 endpoint, DSP environment variables, and AWS instructions do not start,
> configure, or expose `blender-body-mesh` (`bodymesh-server`). The deployed MCP endpoint therefore
> exposes the 49 DSP tools across 11 packs, not the six Blender body-generation tools.

The image installs the whole Python project, so the `bodymesh_server` module and `bodymesh-server`
console entrypoint may be present incidentally. The container command never launches that entrypoint,
and the supported external body-generation runtime is absent.

## Why the body-mesh MCP is not in this deployment

`blender-body-mesh` is currently a separate Windows-local stdio process. Although its Python bridge
code is installed with the project, the Blender 4.2 application, MPFB extension data, sex-specific
engine skeleton files, `character_bake_cli`, and private input staging are not bundled or configured
in the DSP container.

Running body generation on EC2 would be a separate deployment project, not an extra flag on this
image. It would require an appropriate Blender/MPFB host image or Windows instance, separately built
engine tools, authenticated private-photo upload/staging, persistent job storage, quotas, process
isolation, and a deliberately designed remote transport. Until that boundary exists, keep
`blender-body-mesh` local and sync only selected generated artifacts into the DSP deployment.

## Build and push the image

```bash
docker build -t dsp-geometry-engine .
# compose declares DSP_AUTH_TOKEN as required; without it the run aborts before the container starts.
DSP_AUTH_TOKEN=$(openssl rand -hex 32) docker compose up   # MCP endpoint on http://127.0.0.1:8000/mcp
```

Push to Amazon ECR:

```bash
aws ecr create-repository --repository-name dsp-geometry-engine --region <region>
aws ecr get-login-password --region <region> \
  | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag dsp-geometry-engine:latest \
  <account>.dkr.ecr.<region>.amazonaws.com/dsp-geometry-engine:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/dsp-geometry-engine:latest
```

## AWS: App Runner (simplest path)

1. Create an App Runner service pointing at the ECR image.
2. Port: 8000. Health check: **TCP only** — an HTTP probe against `/mcp` can never return 2xx
   (the bearer middleware 401s unauthenticated probes, and FastMCP 4xxes plain GETs without MCP
   Accept headers).
3. Set env vars (see reference table below). `DSP_AUTH_TOKEN` (from Secrets Manager) is mandatory:
   without it the HTTP transport exits at startup instead of serving. `DSP_HOST=0.0.0.0` is already
   in the image.
4. Deploy. App Runner gives you TLS and a URL out of the box.

Caveat: App Runner has no persistent volume — `/data` is ephemeral, so telemetry files must be
re-synced after each deploy/restart. Fine for stateless analysis sessions; use Fargate + EFS when
the data volume must persist.

## AWS: ECS Fargate + EFS (persistent data volume)

1. Create an EFS filesystem + access point.
2. Task definition: the ECR image, port mapping 8000, env vars from the table, and an EFS volume
   mounted at `/data`.
3. Run behind an ALB; terminate TLS there and prefer ALB/API Gateway-level auth (OIDC authorizer)
   in front of the bearer token.
4. Sync telemetry into EFS (e.g. a small sync task, DataSync, or `aws s3 sync` from a sidecar or
   scheduled task writing to the mount).

## AWS Lambda: possible, not recommended

Lambda + the Web Adapter can host the stateless HTTP transport, but it is not recommended here:
MCP sessions and streaming responses fit poorly with Lambda's request/response and freeze model,
and `/data` would have to be re-hydrated per invocation. Use a container service instead.

## Any other container host

The image runs unchanged on Google Cloud Run, Azure Container Apps, and Fly.io — only two things
differ per platform: how a persistent volume gets mounted at `/data` (Cloud Run: GCS FUSE volume;
Azure: Azure Files; Fly: Fly Volumes) and how you wire platform auth in front of the HTTP endpoint.
Nothing in the server is AWS-specific: it reads and writes plain files under `DSP_DATA_DIR`.

## `dsp-server` environment variable reference

These variables configure `dsp-server` only. `BODYMESH_*` variables belong to the separate local
`bodymesh-server` registration and have no effect on this container.

| Variable | Default | Meaning |
| --- | --- | --- |
| `DSP_TRANSPORT` | `stdio` | `stdio` (local Claude Code/Desktop) or `streamable-http` (remote/cloud). The Dockerfile sets `streamable-http`. |
| `DSP_HOST` | `127.0.0.1` | HTTP bind host (HTTP transport only). Loopback by default so an accidental HTTP run is not network-reachable, and because the MCP SDK only auto-enables DNS-rebinding protection on loopback hosts. The Dockerfile sets `0.0.0.0` explicitly — in a container the published port, not the bind address, is the boundary. |
| `DSP_PORT` | `8000` | HTTP port (HTTP transport only). |
| `DSP_AUTH_TOKEN` | unset | **Required for `streamable-http`.** Static bearer token enforced by an ASGI middleware in front of FastMCP's app; a mismatch is a 401. Unset or empty and the server exits at startup rather than serving unauthenticated (see `DSP_ALLOW_INSECURE_HTTP`). Minimum-viable auth — front with real platform auth in production. Ignored on `stdio`. |
| `DSP_ALLOW_INSECURE_HTTP` | unset | Escape hatch: exactly `1` accepts an unauthenticated HTTP port (loopback-only development) and logs a warning naming the address. Any other value, including `true`/`yes`, leaves the server fail-closed. Never set it on a deployment. |
| `DSP_DATA_DIR` | `./data` (image: `/data`) | Root for `dumps/`, `cache/`, `plots/`, `logs/`. Mount the persistent volume here. |
| `DSP_ENGINE_ROOT` | a sibling `proje7-engine` directory next to this repo | Root of the private engine checkout used for exe discovery under `build/<preset>/`. The default is derived from this package's own location, so it is a guess, not a machine-specific path. Local only — meaningless in the container, which ships no engine. |
| `DSP_ENGINE_CLI` | unset | Single-path override of engine discovery; a `.py` path runs via the current interpreter (that is how CI points it at `tests/stub_engine.py`). Operator-only — see security. |
| `DSP_TOOLSETS` | unset (= all 11) | Comma list of packs to register: `geometry`, `imaging`, `stats`, `engmath`, `systems`, `ml`, `netqueue`, `os`, `rendering`, `video`, `perceptual`. Unknown names are skipped with a warning. |
| `MPLBACKEND` | `Agg` (set by the image) | Belt-and-braces only. `src/dsp_server/plots.py` calls `matplotlib.use("Agg")` unconditionally before pyplot is imported, so the server is headless-safe even with this unset or set to something else. |

## Cloud analysis-only mode

The container has no engine executable, and that costs exactly one tool. Of the 49 DSP tools, only
`extract_mesh_telemetry` shells out to the C++ engine (`src/dsp_server/toolsets/geometry.py`); the
bridge's feature detection makes it (and the optional render bridge) return a structured no-engine
error instead of crashing. The other 48 are pure Python over files on disk and work unchanged on any
PLY / palette sidecar / PNG / CSV / NPZ present under `/data`. Even the one exception is coverable
without C++: `DSP_ENGINE_CLI=tests/stub_engine.py` points the bridge at the CLI-faithful stub engine
(a synthetic corrugated cylinder), which is exactly what CI does to exercise the whole subprocess
path on ubuntu runners.

Telemetry is produced on the local Windows box (where the exes, GPU, and the character library live)
and synced to the cloud:

```powershell
# local box: publish telemetry
aws s3 sync data/ s3://<bucket>/dsp-data/ --exclude "cache/*"
```

```bash
# cloud side (or an EFS/DataSync job): hydrate the volume
aws s3 sync s3://<bucket>/dsp-data/ /data/
```

There is deliberately no S3 SDK inside the server — filesystem-only I/O keeps every tool
provider-agnostic. fsspec/s3fs is noted as a possible future optional extra if URI-native paths
(`s3://...` passed directly to tools) are ever wanted.

## Security

> **The HTTP transport is fail-closed.** `DSP_TRANSPORT=streamable-http` without `DSP_AUTH_TOKEN`
> raises `SystemExit` before a socket is opened; the process does not start. The only way past that
> is `DSP_ALLOW_INSECURE_HTTP=1`, which serves every tool to anyone who can reach the port and says
> so in a startup warning — for loopback development only, never a deployment. In production put
> real auth in front of the bearer token (ALB OIDC / API Gateway authorizer / platform equivalent)
> and keep the service off the public internet where possible.
>
> **The bind address defaults to loopback.** `DSP_HOST` is `127.0.0.1` unless overridden, so a
> forgotten HTTP run stays on the box; the container image opts into `0.0.0.0` deliberately.
>
> **`DSP_ENGINE_CLI` must never be client-settable.** The server shells out to whatever that path
> points at — it is an operator-level deployment setting (task definition / service config), never
> derived from tool arguments or request data. The same applies to `DSP_ENGINE_ROOT`.
>
> **Tool arguments are file paths.** Every pack reads and writes real files under `DSP_DATA_DIR`
> and accepts caller-supplied paths, so an authenticated client of this endpoint is trusted with the
> container's filesystem. Scope the task role and the mounted volume accordingly.
