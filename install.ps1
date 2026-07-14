# mnema installer for Windows — downloads prebuilt mnema.exe (CLI) + mnema-server.exe (MCP server).
# No Rust toolchain required.
#
#   irm https://raw.githubusercontent.com/MerlijnW70/mnema/main/install.ps1 | iex
#
# Env:
#   MNEMA_BIN_DIR   install directory     (default: %LOCALAPPDATA%\mnema\bin)
#   MNEMA_VERSION   release tag to fetch  (default: latest, e.g. v0.1.0)

$ErrorActionPreference = 'Stop'
$repo = 'MerlijnW70/mnema'

$binDir = if ($env:MNEMA_BIN_DIR) { $env:MNEMA_BIN_DIR } else { Join-Path $env:LOCALAPPDATA 'mnema\bin' }

# Only x86_64 Windows binaries are published; arm64 users can 'cargo install' from source.
$arch = $env:PROCESSOR_ARCHITECTURE
if ($arch -ne 'AMD64') {
    throw "unsupported architecture '$arch' — run: cargo install --git https://github.com/$repo mnema --features mcp"
}
$target = 'x86_64-pc-windows-msvc'

# Resolve the release tag WITHOUT the unauthenticated GitHub API (rate-limited per IP, so it 403s
# behind shared NAT or CI): the /releases/latest URL redirects to /releases/tag/<tag>, so read the
# redirect's Location header. Uses .NET WebRequest so it works on Windows PowerShell 5.1 and 7.
$tag = $env:MNEMA_VERSION
if (-not $tag) {
    try {
        $req = [System.Net.HttpWebRequest]::Create("https://github.com/$repo/releases/latest")
        $req.AllowAutoRedirect = $false
        $req.UserAgent = 'mnema-install'
        $resp = $req.GetResponse()
        $tag = $resp.Headers['Location'] -replace '.*/tag/', ''
        $resp.Close()
    }
    catch {}
}
if (-not $tag) { throw "could not resolve the latest release — set MNEMA_VERSION (e.g. v0.1.0)" }

$asset = "mnema-$tag-$target.zip"
$url = "https://github.com/$repo/releases/download/$tag/$asset"

$tmp = New-Item -ItemType Directory -Path (Join-Path $env:TEMP ("mnema-" + [System.Guid]::NewGuid()))
try {
    $zip = Join-Path $tmp $asset
    Write-Host "Downloading $asset ..."
    Invoke-WebRequest -Uri $url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $tmp -Force

    New-Item -ItemType Directory -Force -Path $binDir | Out-Null
    $src = Join-Path $tmp "mnema-$tag-$target"
    Copy-Item (Join-Path $src 'mnema.exe') $binDir -Force
    Copy-Item (Join-Path $src 'mnema-server.exe') $binDir -Force

    Write-Host "Installed mnema $tag to ${binDir}:"
    Write-Host "  $binDir\mnema.exe        (CLI)"
    Write-Host "  $binDir\mnema-server.exe    (MCP server)"

    # Add to the user PATH if it isn't already there.
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (($userPath -split ';') -notcontains $binDir) {
        [Environment]::SetEnvironmentVariable('Path', "$userPath;$binDir", 'User')
        Write-Host "Added $binDir to your user PATH (restart the shell to pick it up)."
    }

    Write-Host @"

Point your MCP client at the server (it creates + encrypts the store on first use):
  {
    "mcpServers": {
      "mnema": {
        "command": "$binDir\mnema-server.exe",
        "args": ["--path", "$env:USERPROFILE\mnema.store"]
      }
    }
  }
Set MNEMA_KEY to a passphrase, or omit it to use an auto-generated per-store key file.
"@
}
finally {
    Remove-Item -Recurse -Force $tmp
}
