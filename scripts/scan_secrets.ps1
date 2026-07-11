param(
    [switch]$IncludeIgnored
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$secretPatterns = @(
    'sk-[A-Za-z0-9_-]{20,}',
    'AIza[0-9A-Za-z_-]{20,}',
    'AKIA[0-9A-Z]{16}',
    'Bearer\s+[A-Za-z0-9._-]{20,}',
    '(api[_-]?key|token|secret|password)\s*[:=]\s*["''][^"'']{16,}["'']'
)

$excludedPathPattern = '(^|/)(node_modules|dist|build|\.git|\.venv|venv|env|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache)(/|$)'
$excludedFilePattern = '(^|/)(package-lock\.json|.*\.lock)$'
$findArgs = @('--files')
if ($IncludeIgnored) {
    $findArgs += @('--no-ignore')
}

$findArgs += @(
    '--glob', '!**/node_modules/**',
    '--glob', '!**/dist/**',
    '--glob', '!**/build/**',
    '--glob', '!**/.git/**',
    '--glob', '!**/.venv/**',
    '--glob', '!**/venv/**',
    '--glob', '!**/env/**',
    '--glob', '!**/__pycache__/**',
    '--glob', '!**/package-lock.json',
    '--glob', '!**/*.lock'
)

$files = & rg @findArgs
$findings = New-Object System.Collections.Generic.List[string]

foreach ($file in $files) {
    $normalized = $file -replace '\\', '/'
    if ($normalized -match $excludedPathPattern -or $normalized -match $excludedFilePattern) {
        continue
    }
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) {
        continue
    }

    $lineNo = 0
    Get-Content -LiteralPath $file -ErrorAction SilentlyContinue | ForEach-Object {
        $lineNo += 1
        $line = $_
        foreach ($pattern in $secretPatterns) {
            if ($line -match $pattern) {
                $findings.Add("$file`:$lineNo possible secret pattern")
                break
            }
        }
    }
}

if ($findings.Count -gt 0) {
    Write-Host "Possible secrets found. Values are intentionally hidden:" -ForegroundColor Red
    $findings | Sort-Object -Unique | ForEach-Object { Write-Host $_ }
    exit 1
}

Write-Host "Secret scan passed. No high-risk hardcoded secret patterns found." -ForegroundColor Green
