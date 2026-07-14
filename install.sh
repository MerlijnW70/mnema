#!/bin/sh
# mnema installer — downloads prebuilt `mnema` (CLI) and `mnema-server` (MCP server) binaries.
# No Rust toolchain required.
#
#   curl -fsSL https://raw.githubusercontent.com/MerlijnW70/mnema/main/install.sh | sh
#
# Env:
#   MNEMA_BIN_DIR   install directory     (default: $HOME/.local/bin)
#   MNEMA_VERSION   release tag to fetch  (default: latest, e.g. v0.1.0)
set -eu

REPO="MerlijnW70/mnema"
BIN_DIR="${MNEMA_BIN_DIR:-$HOME/.local/bin}"

err() {
	echo "mnema install: $1" >&2
	exit 1
}

command -v curl >/dev/null 2>&1 || err "curl is required"
command -v tar >/dev/null 2>&1 || err "tar is required"

# --- detect the target triple -------------------------------------------------
os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
Linux) os_part="unknown-linux-musl" ;;
Darwin) os_part="apple-darwin" ;;
*) err "unsupported OS '$os' — on Windows use install.ps1, or 'cargo install --git https://github.com/$REPO mnema --features mcp'" ;;
esac
case "$arch" in
x86_64 | amd64) arch_part="x86_64" ;;
arm64 | aarch64) arch_part="aarch64" ;;
*) err "unsupported architecture '$arch'" ;;
esac
target="${arch_part}-${os_part}"

# --- resolve the release tag --------------------------------------------------
# Resolve the latest tag WITHOUT the unauthenticated GitHub API (rate-limited to 60 req/hr per IP,
# so it 403s behind shared NAT or CI): the /releases/latest page redirects to /releases/tag/<tag>,
# so follow the redirect and take the final URL's last path segment.
tag="${MNEMA_VERSION:-}"
if [ -z "$tag" ]; then
	tag="$(curl -fsSLo /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest" |
		sed 's#.*/tag/##')"
fi
[ -n "$tag" ] || err "could not resolve the latest release — set MNEMA_VERSION (e.g. v0.1.0)"

# --- download + extract -------------------------------------------------------
asset="mnema-${tag}-${target}.tar.gz"
url="https://github.com/$REPO/releases/download/${tag}/${asset}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Downloading $asset ..."
curl -fsSL "$url" -o "$tmp/mnema.tar.gz" || err "download failed: $url"

# --- verify against the release's SHA256SUMS ----------------------------------
# The install one-liner must not blindly trust whatever the URL serves. Fetch the release's
# signed-by-GitHub SHA256SUMS and check the download against it; refuse on mismatch or if the
# release predates checksums (v0.1.4+).
if command -v sha256sum >/dev/null 2>&1; then
	sha256() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
	sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
else
	err "need sha256sum or shasum to verify the download"
fi
curl -fsSL "https://github.com/$REPO/releases/download/${tag}/SHA256SUMS" -o "$tmp/SHA256SUMS" ||
	err "could not fetch SHA256SUMS for $tag — refusing to install unverified (releases before v0.1.4 have none; set MNEMA_VERSION=v0.1.4 or later)"
expected="$(grep -F -- "$asset" "$tmp/SHA256SUMS" | awk '{print $1}' | head -1)"
[ -n "$expected" ] || err "SHA256SUMS has no entry for $asset — refusing to install unverified"
actual="$(sha256 "$tmp/mnema.tar.gz")"
[ "$actual" = "$expected" ] ||
	err "checksum mismatch for $asset — refusing to install (expected $expected, got $actual)"
echo "Verified $asset (sha256 OK)."

tar xzf "$tmp/mnema.tar.gz" -C "$tmp"

# --- install ------------------------------------------------------------------
mkdir -p "$BIN_DIR"
src="$tmp/mnema-${tag}-${target}"
install -m 0755 "$src/mnema" "$BIN_DIR/mnema"
install -m 0755 "$src/mnema-server" "$BIN_DIR/mnema-server"

echo "Installed mnema $tag to $BIN_DIR:"
echo "  $BIN_DIR/mnema        (CLI)"
echo "  $BIN_DIR/mnema-server    (MCP server)"

case ":$PATH:" in
*":$BIN_DIR:"*) ;;
*) echo "
NOTE: $BIN_DIR is not on your PATH. Add it, e.g.:
  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.profile" ;;
esac

echo "
Point your MCP client at the server (it creates + encrypts the store on first use):
  {
    \"mcpServers\": {
      \"mnema\": {
        \"command\": \"$BIN_DIR/mnema-server\",
        \"args\": [\"--path\", \"$HOME/mnema.store\"]
      }
    }
  }
Set MNEMA_KEY to a passphrase, or omit it to use an auto-generated per-store key file."
