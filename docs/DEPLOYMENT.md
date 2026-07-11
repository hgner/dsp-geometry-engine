# Deployment

The analysis server is pure Python + scipy. The container image ships no engine binary — see
"Cloud analysis-only mode" below for what that means. Transport in containers is
`streamable-http` (set by the Dockerfile); tool code is identical to the local stdio deployment.

## Build and push the image

```bash
docker build -t dsp-geometry-engine .
docker compose up          # local smoke: MCP endpoint on http://localhost:8000/mcp
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
2. Port: 8000. Health check: TCP (or the MCP endpoint path).
3. Set env vars (see reference table below) — at minimum `DSP_AUTH_TOKEN` (from Secrets Manager).
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

## Environment variable reference

| Variable | Default | Meaning |
| --- | --- | --- |
| `DSP_TRANSPORT` | `stdio` | `stdio` (local Claude Code/Desktop) or `streamable-http` (remote/cloud). The Dockerfile sets `streamable-http`. |
| `DSP_HOST` | `0.0.0.0` | HTTP bind host (HTTP transport only). |
| `DSP_PORT` | `8000` | HTTP port (HTTP transport only). |
| `DSP_AUTH_TOKEN` | unset | Optional static bearer token enforced on the HTTP transport. Minimum-viable auth — front with real platform auth in production. |
| `DSP_DATA_DIR` | `./data` (image: `/data`) | Root for `dumps/`, `cache/`, `plots/`, `logs/`. Mount the persistent volume here. |
| `DSP_ENGINE_ROOT` | `C:/Users/hgner/hakantest/proje7-engine` | Engine checkout used for exe discovery (local only; meaningless in the container). |
| `DSP_ENGINE_CLI` | unset | Single-path override of engine discovery; a `.py` path runs via the current interpreter (CI stub). Operator-only — see security. |
| `DSP_TOOLSETS` | unset (= all) | Comma list of toolset packs to register (`geometry`, `imaging`, future course packs). |
| `MPLBACKEND` | `Agg` (set by image and `.mcp.json`) | Headless matplotlib — required, there is no display. |

## Cloud analysis-only mode

The container has no engine executable, and that is by design: the bridge's feature detection makes
`extract_mesh_telemetry` (and the optional render bridge) return a structured no-engine error
instead of crashing, while the five analysis tools (`analyze_corrugation`,
`compare_geometry_signals`, `localize_defect`, `lbs_differential`, `compare_depth_renders`) work on
any PLY / palette sidecar / PNG files present under `/data`.

Telemetry is produced on the local Windows box (where the exes, GPU, and `D:\` assets live) and
synced to the cloud:

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

> **Never expose the HTTP transport unauthenticated.** At minimum set `DSP_AUTH_TOKEN`; in
> production put real auth in front (ALB OIDC / API Gateway authorizer / platform equivalent) and
> keep the service off the public internet where possible.
>
> **`DSP_ENGINE_CLI` must never be client-settable.** The server shells out to whatever that path
> points at — it is an operator-level deployment setting (task definition / service config), never
> derived from tool arguments or request data. The same applies to `DSP_ENGINE_ROOT`.
