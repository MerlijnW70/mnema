# mnema installer for Windows - downloads prebuilt mnema.exe (CLI) + mnema-server.exe (MCP server).
# No Rust toolchain required.
#
# KEEP THIS FILE PURE ASCII. Windows PowerShell 5.1 reads a BOM-less .ps1 file as ANSI, so a
# UTF-8 em-dash decodes as a cp1252 smart-quote that terminates string literals and the whole
# script fails to parse for anyone who downloads it and runs it as a file (instead of irm | iex).
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
    throw "unsupported architecture '$arch' - run: cargo install --git https://github.com/$repo mnema --features mcp"
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
if (-not $tag) { throw "could not resolve the latest release - set MNEMA_VERSION (e.g. v0.1.0)" }

$asset = "mnema-$tag-$target.zip"
$url = "https://github.com/$repo/releases/download/$tag/$asset"

$tmp = New-Item -ItemType Directory -Path (Join-Path $env:TEMP ("mnema-" + [System.Guid]::NewGuid()))
try {
    $zip = Join-Path $tmp $asset
    Write-Host "Downloading $asset ..."
    Invoke-WebRequest -Uri $url -OutFile $zip

    # Verify against the release's SHA256SUMS - don't blindly trust what the URL served. Refuse on
    # mismatch, or if the release predates checksums (v0.1.4+).
    $sumsFile = Join-Path $tmp 'SHA256SUMS'
    try {
        Invoke-WebRequest -Uri "https://github.com/$repo/releases/download/$tag/SHA256SUMS" -OutFile $sumsFile
    } catch {
        throw "could not fetch SHA256SUMS for $tag - refusing to install unverified (releases before v0.1.4 have none; set MNEMA_VERSION to v0.1.4 or later)"
    }
    $line = Select-String -Path $sumsFile -SimpleMatch $asset | Select-Object -First 1
    if (-not $line) { throw "SHA256SUMS has no entry for $asset - refusing to install unverified" }
    $expected = ($line.Line -replace '\s.*', '').ToLower()
    $actual = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
    if ($actual -ne $expected) {
        throw "checksum mismatch for $asset - refusing to install (expected $expected, got $actual)"
    }
    Write-Host "Verified $asset (sha256 OK)."

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

    # Forward slashes: backslashes are escape characters in JSON, so a pasted "C:\Users\..." is
    # invalid. JSON with forward slashes works fine in Windows paths.
    $binJson = $binDir -replace '\\', '/'
    $storeJson = "$env:USERPROFILE\mnema.store" -replace '\\', '/'
    Write-Host @"

Point your MCP client at the server (it creates + encrypts the store on first use):
  {
    "mcpServers": {
      "mnema": {
        "command": "$binJson/mnema-server.exe",
        "args": ["--path", "$storeJson"]
      }
    }
  }
Set MNEMA_KEY to a passphrase, or omit it to use an auto-generated per-store key file.
"@
}
finally {
    Remove-Item -Recurse -Force $tmp
}
