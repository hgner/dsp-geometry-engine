# Full local verification lane: format + lint + tests, then the C++ engine smoke
# (only if an engine exe resolves), then a docker compose smoke (only if docker is
# present). Any failing step exits 1 immediately.
#
# PowerShell 5.1 compatible (no &&, no ternary).
[CmdletBinding()]
param()

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Output "verify-local: uv run ruff format ."
uv run ruff format .
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Output "verify-local: uv run ruff check ."
uv run ruff check .
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Output "verify-local: uv run pytest"
uv run pytest
if ($LASTEXITCODE -ne 0) { exit 1 }

# --- Engine smoke: only when the exe resolves (same rules as verify-engine.ps1) ---
$engineRoot = $env:DSP_ENGINE_ROOT
if ([string]::IsNullOrWhiteSpace($engineRoot)) {
  # Fallback: an engine checkout sitting next to this repository's own directory.
  $engineRoot = Join-Path (Split-Path -Parent $repoRoot) "proje7-engine"
}
$exe = $env:DSP_ENGINE_CLI
if ([string]::IsNullOrWhiteSpace($exe)) {
  $candidates = @(
    (Join-Path $engineRoot "build\windows-msvc-static-md-release\layered_field_dump_cli.exe"),
    (Join-Path $engineRoot "build\windows-msvc-rtx\layered_field_dump_cli.exe")
  )
  foreach ($c in $candidates) {
    if (Test-Path -LiteralPath $c) { $exe = $c; break }
  }
}
if (-not [string]::IsNullOrWhiteSpace($exe) -and (Test-Path -LiteralPath $exe)) {
  Write-Output "verify-local: engine exe resolved - running verify-engine.ps1"
  & (Join-Path $PSScriptRoot "verify-engine.ps1")
  if ($LASTEXITCODE -ne 0) { exit 1 }
} else {
  Write-Output "verify-local: engine exe not found - SKIPPED verify-engine.ps1"
}

# --- Container smoke: only when docker is installed --------------------------------
$docker = Get-Command docker -ErrorAction SilentlyContinue
if ($docker) {
  # compose.yaml declares DSP_AUTH_TOKEN as mandatory (${DSP_AUTH_TOKEN:?set a token})
  # because the HTTP transport is fail-closed. Supply a throwaway token for the smoke
  # so interpolation resolves; a real deployment sets a long random one.
  if ([string]::IsNullOrWhiteSpace($env:DSP_AUTH_TOKEN)) {
    $env:DSP_AUTH_TOKEN = [guid]::NewGuid().ToString('N')
  }

  Write-Output "verify-local: docker compose build"
  docker compose build
  if ($LASTEXITCODE -ne 0) { exit 1 }

  Write-Output "verify-local: docker compose up -d"
  docker compose up -d
  if ($LASTEXITCODE -ne 0) { exit 1 }

  Start-Sleep -Seconds 5
  $running = docker compose ps -q
  if (-not $running) {
    docker compose logs
    docker compose down
    Write-Output "verify-local: container exited during smoke - FAIL"
    exit 1
  }
  Write-Output "verify-local: container up on http://127.0.0.1:8000/mcp - tearing down"
  docker compose down
  if ($LASTEXITCODE -ne 0) { exit 1 }
} else {
  Write-Output "verify-local: docker not found - SKIPPED container smoke"
}

Write-Output "verify-local: OK"
exit 0
