# Requires Python 3.9+, Rust/MSVC, CMake, LLVM/libclang, and WebView2.
# All arguments are forwarded, e.g. -File install/install.ps1 --cuda --jobs 2.
$ErrorActionPreference = 'Stop'
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command python3 -ErrorAction SilentlyContinue }
if (-not $python) {
    throw 'Python 3.9+ is required. Install Python explicitly, then rerun this installer.'
}
& $python.Source (Join-Path $PSScriptRoot 'source_install.py') @args
exit $LASTEXITCODE
