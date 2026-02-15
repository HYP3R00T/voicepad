# Check if commitizen is installed
if (-not (Get-Command cz -ErrorAction SilentlyContinue)) {
    Write-Host "Installing commitizen via uv tool..."
    uv tool install commitizen
}

# Install pre-commit hook if not present
if (-not (Test-Path ".git\hooks\pre-commit")) {
    Write-Host "Installing pre-commit hook..."
    pre-commit install
}

# Install commit-msg hook if not present
if (-not (Test-Path ".git\hooks\commit-msg")) {
    Write-Host "Installing commit-msg hook..."
    pre-commit install --hook-type commit-msg
}
