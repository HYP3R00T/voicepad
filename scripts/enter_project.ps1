Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Get-Command cz -ErrorAction SilentlyContinue)) {
	uv tool install commitizen
	if (-not (Get-Command cz -ErrorAction SilentlyContinue)) {
		throw 'Failed to install commitizen'
	}
}

if (Get-Command prek -ErrorAction SilentlyContinue) {
	$pre_commit_hook = Join-Path '.git/hooks' 'pre-commit'
	if (-not (Test-Path $pre_commit_hook) -or -not (Select-String -Path $pre_commit_hook -Pattern 'prek' -Quiet -ErrorAction SilentlyContinue)) {
		prek install --overwrite | Out-Null
	}

	$commit_msg_hook = Join-Path '.git/hooks' 'commit-msg'
	if (-not (Test-Path $commit_msg_hook) -or -not (Select-String -Path $commit_msg_hook -Pattern 'prek' -Quiet -ErrorAction SilentlyContinue)) {
		prek install --hook-type commit-msg --overwrite | Out-Null
	}
}
