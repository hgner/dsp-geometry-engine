# Register the DSP and Blender body-mesh MCP servers in Claude Desktop.
# Creates %APPDATA%\Claude if missing and MERGES both entries into
# claude_desktop_config.json — sibling servers and other top-level keys are never
# clobbered. Desktop does not inherit the full user PATH, so `command` is the
# absolute uv.exe path resolved here. Inert until Claude Desktop is installed.
#
# PowerShell 5.1 compatible (no &&, no ternary).
[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"

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

# Mirror of .mcp.json, with `command` resolved to the absolute uv.exe.
$server = [pscustomobject]@{
  command = $uvPath
  args    = @("--directory", "C:/Users/hgner/hakantest/proje10", "run", "dsp-server")
  env     = [pscustomobject]@{
    MPLBACKEND      = "Agg"
    DSP_ENGINE_ROOT = "C:/Users/hgner/hakantest/proje7-engine"
    DSP_DATA_DIR    = "C:/Users/hgner/hakantest/proje10/data"
  }
}

$bodyServer = [pscustomobject]@{
  command = $uvPath
  args    = @("--directory", "C:/Users/hgner/hakantest/proje10", "run", "bodymesh-server")
  env     = [pscustomobject]@{
    MPLBACKEND                   = "Agg"
    BODYMESH_BLENDER_SHORTCUT    = "C:/Users/hgner/Desktop/Blender 4.2.lnk"
    BODYMESH_BLENDER_EXE         = "C:/Program Files/Blender Foundation/Blender 4.2/blender.exe"
    BODYMESH_MPFB_ROOT           = "C:/Users/hgner/AppData/Roaming/Blender Foundation/Blender/4.2/extensions/user_default/mpfb"
    BODYMESH_MBLAB_ROOT          = "C:/Users/hgner/AppData/Roaming/Blender Foundation/Blender/4.2/scripts/addons/MB-Lab-master"
    BODYMESH_DATA_DIR            = "C:/Users/hgner/hakantest/proje10/data/bodymesh"
    BODYMESH_INPUT_ROOTS         = "C:/Users/hgner/hakantest/proje10/data;C:/Users/hgner/Desktop"
  }
}

# -Force adds or replaces ONLY our entry; every other server/key is left untouched.
$config.mcpServers | Add-Member -MemberType NoteProperty -Name "dsp-geometry-engine" -Value $server -Force
$config.mcpServers | Add-Member -MemberType NoteProperty -Name "blender-body-mesh" -Value $bodyServer -Force

$json = $config | ConvertTo-Json -Depth 16
[System.IO.File]::WriteAllText($configPath, $json, (New-Object System.Text.UTF8Encoding($false)))

Write-Output "wrote $configPath (UTF-8, no BOM)"
Write-Output "registered mcpServers.dsp-geometry-engine:"
Write-Output "  command: $uvPath"
Write-Output "  args:    --directory C:/Users/hgner/hakantest/proje10 run dsp-server"
Write-Output ("  env:     MPLBACKEND=Agg DSP_ENGINE_ROOT=C:/Users/hgner/hakantest/proje7-engine " +
  "DSP_DATA_DIR=C:/Users/hgner/hakantest/proje10/data")
Write-Output "registered mcpServers.blender-body-mesh:"
Write-Output "  command: $uvPath"
Write-Output "  args:    --directory C:/Users/hgner/hakantest/proje10 run bodymesh-server"
Write-Output "  backend: Blender 4.2 + MPFB 2.0.x (local stdio only)"
Write-Output "NOTE: restart Claude Desktop for the new server to appear."
exit 0
