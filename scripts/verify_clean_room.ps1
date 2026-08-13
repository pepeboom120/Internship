$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$verifyEnvironment = Join-Path $repoRoot '.verify-venv'
$bundledPython = 'C:\Users\twpow\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'

if (-not (Test-Path $verifyEnvironment)) {
    & $bundledPython -m venv $verifyEnvironment
}
$verifyPython = Join-Path $verifyEnvironment 'Scripts\python.exe'
& $verifyPython -m pip install -e "$repoRoot[dev]"
if ($LASTEXITCODE -ne 0) { throw "Clean-room package installation failed." }
& $verifyPython -m ruff check "$repoRoot\src" "$repoRoot\tests"
if ($LASTEXITCODE -ne 0) { throw "Clean-room Ruff check failed." }
& $verifyPython -m pytest -q --basetemp="$repoRoot\.verify-pytest-tmp" "$repoRoot\tests"
if ($LASTEXITCODE -ne 0) { throw "Clean-room pytest failed." }
Write-Host 'Clean-room package installation, lint, and tests passed.'
