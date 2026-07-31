# Local C++ engine smoke verification (plan Verification 2). Runs the prebuilt
# layered_field_dump_cli: one rest dump + two posed procedural dumps, then (post-M5
# patch, feature-detected via --help) the rigged3 --character dump with a palette
# sidecar. Rest iso-surface vertex counts are NEVER compared against posed counts —
# only posed-vs-posed (different pipelines). Final line: VERIFY_OK or
# VERIFY_FAIL: <reason>; exit 0/1.
#
# PowerShell 5.1 compatible (no &&, no ternary).
[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$verifyDir = Join-Path $repoRoot "data\verify"
New-Item -ItemType Directory -Force -Path $verifyDir | Out-Null

function Fail([string] $Reason) {
  Write-Output "VERIFY_FAIL: $Reason"
  exit 1
}

# --- Resolve engine root + CLI (env override, then the two preset candidates) ---
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
if ([string]::IsNullOrWhiteSpace($exe) -or -not (Test-Path -LiteralPath $exe)) {
  Fail "engine CLI not found (set DSP_ENGINE_CLI or run scripts\build-engine.ps1)"
}

# cwd = engine root only if it exists (mirrors the bridge's rule), else repo root.
$workDir = $repoRoot
if (Test-Path -LiteralPath $engineRoot) { $workDir = $engineRoot }

Write-Output "verify-engine: exe=$exe"
Write-Output "verify-engine: cwd=$workDir"

# --- Child runner: 600 s timeout, stdout/stderr captured to temp files ----------
function Invoke-EngineCli([string[]] $CliArgs) {
  $outFile = [System.IO.Path]::GetTempFileName()
  $errFile = [System.IO.Path]::GetTempFileName()
  $quoted = @()
  foreach ($a in $CliArgs) {
    if ($a -match '\s') { $quoted += ('"' + $a + '"') } else { $quoted += $a }
  }
  $proc = Start-Process -FilePath $exe -ArgumentList ($quoted -join ' ') `
    -WorkingDirectory $workDir -NoNewWindow -PassThru `
    -RedirectStandardOutput $outFile -RedirectStandardError $errFile
  $null = $proc.Handle  # PS 5.1: cache the handle or ExitCode reads back $null after exit
  $finished = $proc.WaitForExit(600000)
  if (-not $finished) {
    try { $proc.Kill() } catch {}
    try { Remove-Item $outFile, $errFile -Force -ErrorAction Stop } catch {}
    return New-Object psobject -Property @{ TimedOut = $true; ExitCode = -1; StdOut = ""; StdErr = "" }
  }
  $proc.WaitForExit()  # second, untimed wait flushes ExitCode reliably on 5.1
  $stdout = ""
  $stderr = ""
  if (Test-Path -LiteralPath $outFile) { $stdout = [System.IO.File]::ReadAllText($outFile) }
  if (Test-Path -LiteralPath $errFile) { $stderr = [System.IO.File]::ReadAllText($errFile) }
  try { Remove-Item $outFile, $errFile -Force -ErrorAction Stop } catch {}
  return New-Object psobject -Property @{
    TimedOut = $false; ExitCode = $proc.ExitCode; StdOut = $stdout; StdErr = $stderr
  }
}

function Get-PlyVertexCount([string] $PlyPath) {
  $lines = Get-Content -LiteralPath $PlyPath -TotalCount 200
  foreach ($line in $lines) {
    if ($line -match '^element vertex (\d+)') { return [int]$Matches[1] }
    if ($line -match '^end_header') { break }
  }
  return -1
}

function Get-BoneMapCount([string] $StdErrText) {
  foreach ($line in ($StdErrText -split "`r?`n")) {
    if ($line -match 'bone-map:') {
      return ([regex]::Matches($line, '\b\d+=')).Count
    }
  }
  return -1
}

# --- 1) Rest dump ----------------------------------------------------------------
$restPly = Join-Path $verifyDir "rest_male.ply"
Write-Output "verify-engine: rest dump -> $restPly"
$r = Invoke-EngineCli @("--sex", "male", "--out", $restPly)
if ($r.TimedOut) { Fail "rest dump timed out after 600 s" }
if ($r.ExitCode -ne 0) { Fail "rest dump exited $($r.ExitCode)" }
if (-not (Test-Path -LiteralPath $restPly)) { Fail "rest dump wrote no file" }
if ((Get-Item -LiteralPath $restPly).Length -le 0) { Fail "rest dump file is empty" }
$restVerts = Get-PlyVertexCount $restPly
if ($restVerts -le 1000) { Fail "rest dump vertex count $restVerts (expected > 1000)" }
Write-Output "verify-engine: rest dump OK ($restVerts vertices)"

# --- 2) Two posed procedural dumps: exit 0, bone-map present, EQUAL vertex counts --
$poseClips = @("clip-cin-stand-attention", "clip-cin-walktalk")
$posedCounts = @{}
foreach ($clip in $poseClips) {
  $ply = Join-Path $verifyDir ("posed_" + $clip + ".ply")
  Write-Output "verify-engine: posed dump ($clip) -> $ply"
  $r = Invoke-EngineCli @("--sex", "male", "--pose", $clip, "--out", $ply)
  if ($r.TimedOut) { Fail "posed dump ($clip) timed out after 600 s" }
  if ($r.ExitCode -ne 0) { Fail "posed dump ($clip) exited $($r.ExitCode)" }
  if ($r.StdErr -notmatch 'bone-map:') { Fail "posed dump ($clip) stderr missing 'bone-map:'" }
  if (-not (Test-Path -LiteralPath $ply)) { Fail "posed dump ($clip) wrote no file" }
  $count = Get-PlyVertexCount $ply
  if ($count -le 0) { Fail "posed dump ($clip) has no 'element vertex' header line" }
  $posedCounts[$clip] = $count
  Write-Output "verify-engine: posed dump ($clip) OK ($count vertices)"
}
if ($posedCounts[$poseClips[0]] -ne $posedCounts[$poseClips[1]]) {
  $a = $posedCounts[$poseClips[0]]
  $b = $posedCounts[$poseClips[1]]
  Fail "posed vertex counts differ: $($poseClips[0])=$a vs $($poseClips[1])=$b"
}

# --- 3) Feature-detect the M5 patch (--character in --help), then rigged3 + palette --
$help = Invoke-EngineCli @("--help")
$helpText = $help.StdOut + "`n" + $help.StdErr
if ($helpText -match '--character') {
  # Optional stage: a baked character JSON from the operator's own asset library.
  # Point DSP_VERIFY_CHARACTER at one to enable it; unset means skip, not fail.
  $asset = $env:DSP_VERIFY_CHARACTER
  if (-not [string]::IsNullOrWhiteSpace($asset) -and (Test-Path -LiteralPath $asset)) {
    $rigPly = Join-Path $verifyDir "rigged3_walktalk.ply"
    $palette = Join-Path $verifyDir "rigged3_walktalk.palette.json"
    Write-Output "verify-engine: rigged3 posed dump -> $rigPly (+palette)"
    $r = Invoke-EngineCli @(
      "--character", $asset, "--sex", "male", "--pose", "clip-cin-walktalk",
      "--out", $rigPly, "--palette-out", $palette
    )
    if ($r.TimedOut) { Fail "rigged3 dump timed out after 600 s" }
    if ($r.ExitCode -ne 0) { Fail "rigged3 dump exited $($r.ExitCode)" }
    if ($r.StdErr -notmatch 'bone-map:') { Fail "rigged3 dump stderr missing 'bone-map:'" }
    $boneMapCount = Get-BoneMapCount $r.StdErr
    if ($boneMapCount -le 0) { Fail "rigged3 dump bone-map line has no entries" }
    if (-not (Test-Path -LiteralPath $palette)) { Fail "palette sidecar not written: $palette" }
    $pal = $null
    try { $pal = (Get-Content -LiteralPath $palette -Raw) | ConvertFrom-Json } catch {}
    if ($null -eq $pal) { Fail "palette sidecar is not valid JSON: $palette" }
    if ([int]$pal.boneCount -ne $boneMapCount) {
      Fail "palette boneCount $($pal.boneCount) != bone-map entries $boneMapCount"
    }
    Write-Output "verify-engine: rigged3 palette OK (boneCount=$($pal.boneCount))"
  } else {
    Write-Output "WARNING: no character asset (set DSP_VERIFY_CHARACTER to a baked JSON) - skipping --character verification"
  }
} else {
  Write-Output "verify-engine: binary has no --character flag (pre-M5 patch) - skipping rigged3 check"
}

Write-Output "VERIFY_OK"
exit 0
