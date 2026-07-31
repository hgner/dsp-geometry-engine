# Register the DSP and Blender body-mesh MCP servers in Claude Desktop.
# Creates %APPDATA%\Claude if missing and MERGES both entries into
# claude_desktop_config.json — sibling servers and other top-level keys are never
# clobbered. Desktop does not inherit the full user PATH, so `command` is the
# absolute uv.exe path resolved here. Inert until Claude Desktop is installed.
#
# Every path written into the config is DERIVED, never a literal: the repo root comes
# from $PSScriptRoot (no cwd fallback — Desktop launches servers from an arbitrary
# directory), and the optional engine/Blender/MPFB paths are discovered on this
# machine. Anything that cannot be discovered is left unset so the in-code defaults in
# src/dsp_server/config.py and src/bodymesh_server/config.py apply, and the omission is
# reported at the end.
#
# PowerShell 5.1 compatible (no &&, no ternary).
[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"

function ConvertTo-ForwardSlash {
  param([string]$Path)
  return ($Path -replace '\\', '/')
}

# Blender: an explicit env override wins, otherwise the newest "Blender <ver>" under
# Program Files. Always blender.exe — never blender-launcher.exe.
function Resolve-BlenderExe {
  if ($env:BODYMESH_BLENDER_EXE -and (Test-Path -LiteralPath $env:BODYMESH_BLENDER_EXE)) {
    return (Resolve-Path -LiteralPath $env:BODYMESH_BLENDER_EXE).Path
  }
  $programRoots = @(
    [Environment]::GetEnvironmentVariable("ProgramFiles"),
    [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
  ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
  foreach ($programRoot in $programRoots) {
    $base = Join-Path $programRoot "Blender Foundation"
    if (-not (Test-Path -LiteralPath $base)) { continue }
    $dirs = Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue |
      Sort-Object Name -Descending
    foreach ($dir in $dirs) {
      $exe = Join-Path $dir.FullName "blender.exe"
      if (Test-Path -LiteralPath $exe) { return $exe }
    }
  }
  return $null
}

# MPFB 2.0.x extension directory: identified by its blender_manifest.toml.
function Resolve-MpfbRoot {
  $candidates = @()
  if (-not [string]::IsNullOrWhiteSpace($env:BODYMESH_MPFB_ROOT)) { $candidates += $env:BODYMESH_MPFB_ROOT }
  $base = Join-Path $env:APPDATA "Blender Foundation\Blender"
  if (Test-Path -LiteralPath $base) {
    $versions = Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue |
      Sort-Object Name -Descending
    foreach ($version in $versions) {
      $candidates += (Join-Path $version.FullName "extensions\user_default\mpfb")
    }
  }
  foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath (Join-Path $candidate "blender_manifest.toml")) {
      return (Resolve-Path -LiteralPath $candidate).Path
    }
  }
  return $null
}

# The private C++ engine checkout: env override, else a sibling of this repo.
function Resolve-EngineRoot {
  param([string]$RepoRoot)
  if ($env:DSP_ENGINE_ROOT -and (Test-Path -LiteralPath $env:DSP_ENGINE_ROOT)) {
    return (Resolve-Path -LiteralPath $env:DSP_ENGINE_ROOT).Path
  }
  $sibling = Join-Path (Split-Path -Parent $RepoRoot) "proje7-engine"
  if (Test-Path -LiteralPath $sibling) { return (Resolve-Path -LiteralPath $sibling).Path }
  return $null
}

# Engine-canonical skeleton assets: only accepted when both sexes are present.
function Resolve-SkeletonDir {
  param([string]$RepoRoot)
  $candidates = @()
  if (-not [string]::IsNullOrWhiteSpace($env:BODYMESH_ENGINE_SKELETON_DIR)) {
    $candidates += $env:BODYMESH_ENGINE_SKELETON_DIR
  }
  $candidates += (Join-Path (Split-Path -Parent $RepoRoot) "proje8\scripts\blender")
  foreach ($candidate in $candidates) {
    $male = Join-Path $candidate "skeleton_male.json"
    $female = Join-Path $candidate "skeleton_female.json"
    if ((Test-Path -LiteralPath $male) -and (Test-Path -LiteralPath $female)) {
      return (Resolve-Path -LiteralPath $candidate).Path
    }
  }
  return $null
}

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot "pyproject.toml"))) {
  Write-Output "REGISTER_FAIL: $repoRoot has no pyproject.toml - run this script from its place in the repo"
  exit 1
}
$repoRootFwd = ConvertTo-ForwardSlash $repoRoot

$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
  Write-Output "REGISTER_FAIL: uv not found on PATH - install uv first (https://docs.astral.sh/uv/)"
  exit 1
}
$uvPath = $uvCmd.Source

$claudeDir = Join-Path $env:APPDATA "Claude"
if (-not (Test-Path -LiteralPath $claudeDir)) {
  New-Item -ItemType Directory -Force -Path $claudeDir | Out-Null
  Write-Output "created $claudeDir"
}
$configPath = Join-Path $claudeDir "claude_desktop_config.json"

# Load the existing config if present; a corrupt file is a hard stop (never clobber).
$config = $null
if (Test-Path -LiteralPath $configPath) {
  $raw = Get-Content -LiteralPath $configPath -Raw
  if (-not [string]::IsNullOrWhiteSpace($raw)) {
    try {
      $config = $raw | ConvertFrom-Json
    } catch {
      Write-Output "REGISTER_FAIL: $configPath exists but is not valid JSON - fix it by hand first"
      exit 1
    }
  }
}
if ($null -eq $config) { $config = New-Object psobject }

if (-not ($config.PSObject.Properties.Name -contains "mcpServers") -or $null -eq $config.mcpServers) {
  $config | Add-Member -MemberType NoteProperty -Name "mcpServers" -Value (New-Object psobject) -Force
}

$notes = @()

# --- dsp-geometry-engine: 49 tools, only extract_mesh_telemetry needs the engine ----
# DSP_DATA_DIR is pinned under the repo instead of relying on Path.cwd() in
# src/dsp_server/config.py, so telemetry never lands in Desktop's launch directory.
$dspEnv = [ordered]@{
  DSP_DATA_DIR = "$repoRootFwd/data"
}
$engineRoot = Resolve-EngineRoot -RepoRoot $repoRoot
if ($engineRoot) {
  $dspEnv["DSP_ENGINE_ROOT"] = (ConvertTo-ForwardSlash $engineRoot)
} else {
  $notes += "DSP_ENGINE_ROOT unset - proje7-engine checkout not found; extract_mesh_telemetry is unavailable, the other 48 tools are not affected"
}

$server = [pscustomobject]@{
  command = $uvPath
  args    = @("--directory", $repoRootFwd, "run", "dsp-server")
  env     = [pscustomobject]$dspEnv
}

# --- blender-body-mesh: 6 tools, local stdio only ----------------------------------
# BODYMESH_INPUT_ROOTS is the image allowlist enforced at the Blender process boundary;
# it stays pinned to the job inbox rather than a broad user directory.
$bodyEnv = [ordered]@{
  BODYMESH_DATA_DIR    = "$repoRootFwd/data/bodymesh"
  BODYMESH_INPUT_ROOTS = "$repoRootFwd/data/bodymesh/inbox"
}
$blenderExe = Resolve-BlenderExe
if ($blenderExe) {
  $bodyEnv["BODYMESH_BLENDER_EXE"] = (ConvertTo-ForwardSlash $blenderExe)
} else {
  $notes += "BODYMESH_BLENDER_EXE unset - no blender.exe under Program Files\Blender Foundation; install Blender 4.2 or set it by hand before using the body-mesh tools"
}
$mpfbRoot = Resolve-MpfbRoot
if ($mpfbRoot) {
  $bodyEnv["BODYMESH_MPFB_ROOT"] = (ConvertTo-ForwardSlash $mpfbRoot)
} else {
  $notes += "BODYMESH_MPFB_ROOT unset - no MPFB extension with blender_manifest.toml under %APPDATA%\Blender Foundation\Blender; generation stays blocked until MPFB 2.0.x is installed"
}
$skeletonDir = Resolve-SkeletonDir -RepoRoot $repoRoot
if ($skeletonDir) {
  $bodyEnv["BODYMESH_ENGINE_SKELETON_DIR"] = (ConvertTo-ForwardSlash $skeletonDir)
} else {
  $notes += "BODYMESH_ENGINE_SKELETON_DIR unset - skeleton_male.json/skeleton_female.json not found; the engine retarget/bake step is unavailable"
}
if ($engineRoot) {
  $bakeExe = Join-Path $engineRoot "build\windows-msvc-static-md-release\character_bake_cli.exe"
  if (Test-Path -LiteralPath $bakeExe) {
    $bodyEnv["BODYMESH_CHARACTER_BAKE_EXE"] = (ConvertTo-ForwardSlash $bakeExe)
  } else {
    $notes += "BODYMESH_CHARACTER_BAKE_EXE unset - $bakeExe is missing; build the engine preset first"
  }
}

$bodyServer = [pscustomobject]@{
  command = $uvPath
  args    = @("--directory", $repoRootFwd, "run", "bodymesh-server")
  env     = [pscustomobject]$bodyEnv
}

# -Force adds or replaces ONLY our entry; every other server/key is left untouched.
$config.mcpServers | Add-Member -MemberType NoteProperty -Name "dsp-geometry-engine" -Value $server -Force
$config.mcpServers | Add-Member -MemberType NoteProperty -Name "blender-body-mesh" -Value $bodyServer -Force

$json = $config | ConvertTo-Json -Depth 16
[System.IO.File]::WriteAllText($configPath, $json, (New-Object System.Text.UTF8Encoding($false)))

Write-Output "wrote $configPath (UTF-8, no BOM)"
Write-Output "repo root: $repoRootFwd (derived from this script's location)"
Write-Output "registered mcpServers.dsp-geometry-engine:"
Write-Output "  command: $uvPath"
Write-Output "  args:    --directory $repoRootFwd run dsp-server"
Write-Output ("  env:     " + (($dspEnv.Keys | ForEach-Object { "$_=$($dspEnv[$_])" }) -join " "))
Write-Output "registered mcpServers.blender-body-mesh:"
Write-Output "  command: $uvPath"
Write-Output "  args:    --directory $repoRootFwd run bodymesh-server"
Write-Output ("  env:     " + (($bodyEnv.Keys | ForEach-Object { "$_=$($bodyEnv[$_])" }) -join " "))
Write-Output "  backend: Blender 4.2 + MPFB 2.0.x (local stdio only)"
foreach ($note in $notes) {
  Write-Output "NOTE: $note"
}
Write-Output "NOTE: restart Claude Desktop for the new server to appear."
exit 0
