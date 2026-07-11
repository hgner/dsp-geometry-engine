# Thin delegator to proje8's engine build script (plan M5). Rebuilds
# layered_field_dump_cli in proje7-engine after the --character/--palette-out patch.
#
#   powershell -ExecutionPolicy Bypass -File scripts\build-engine.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\build-engine.ps1 -Preset windows-msvc-rtx
#
# PowerShell 5.1 compatible (no &&, no ternary).
[CmdletBinding()]
param(
  [string] $Preset = "windows-msvc-static-md-release",
  [string] $Target = "layered_field_dump_cli"
)
$ErrorActionPreference = "Stop"

$delegate = $env:DSP_ENGINE_BUILD_SCRIPT
if ([string]::IsNullOrWhiteSpace($delegate)) {
  $delegate = "C:\Users\hgner\hakantest\proje8\scripts\build-engine-cli.ps1"
}
if (-not (Test-Path -LiteralPath $delegate)) {
  Write-Output "BUILD_FAIL: delegate build script not found: $delegate (set DSP_ENGINE_BUILD_SCRIPT)"
  exit 2
}

$repoDir = $env:DSP_ENGINE_ROOT
if ([string]::IsNullOrWhiteSpace($repoDir)) {
  $repoDir = "C:\Users\hgner\hakantest\proje7-engine"
}

if ($Preset -eq "windows-msvc-rtx") {
  # FEAT-CG-2 stale-sibling policy: the rtx preset hosts several tool exes linked
  # against the same core objects. Omit -Target so the delegate's per-preset default
  # relinks the WHOLE rtx tool set — naming a single target here would leave sibling
  # exes (e.g. i2v_beauty_dxr_cli) silently linked against stale objects.
  & $delegate -RepoDir $repoDir -Preset $Preset
} else {
  & $delegate -RepoDir $repoDir -Preset $Preset -Target $Target
}
exit $LASTEXITCODE
