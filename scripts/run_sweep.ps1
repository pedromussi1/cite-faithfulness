# Controlled-study sweep: retrieval config x model size.
#
# For each (retrieval config) x (Ollama model) cell, this launches a fresh
# PaperPal uvicorn with the right env vars, waits for /healthz, runs the
# citeval faithfulness eval against it, then kills the backend. Results land
# under cite-faithfulness/runs/<config>-<model-tag>/. Mirrors PaperPal's own
# backend/eval/run_ablation.ps1 (same restart-per-config pattern), extended
# with the model-size axis and pointed at citeval.run_faithfulness.
#
# Usage (from anywhere):
#   powershell -File "d:\AI Projects\cite-faithfulness\scripts\run_sweep.ps1"
#
# Prereqs: Ollama running with the models pulled (ollama pull llama3.2:3b /
# llama3.1:8b); PaperPal backend installed in its .venv; cite-faithfulness
# installed with the [nli] extra. 4 configs x 2 models x ~90 s ~= 12-15 min.

param(
    # By default, cells that already produced a summary.json are skipped
    # (resume). Pass -Force to re-run every cell from scratch.
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# --- paths (edit if your layout differs) ---
$paperpalRoot = "d:\AI Projects\PaperPal\backend"
$paperpalPy   = Join-Path $paperpalRoot ".venv\Scripts\python.exe"
$citeRoot     = "d:\AI Projects\cite-faithfulness"
$citePy       = Join-Path $citeRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $citePy)) { $citePy = "python" }  # fall back to PATH python

$nli = "cross-encoder/nli-deberta-v3-base"
$reranker = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Model-size axis. The tag becomes part of the run name (no ':' in dir names).
$models = @(
    @{ model = "llama3.2:3b"; tag = "3b" },
    @{ model = "llama3.1:8b"; tag = "8b" }
)

# Retrieval-config axis (matches PaperPal's Settings env vars).
$configs = @(
    @{ name = "dense";         hybrid = "false"; reranker = "" },
    @{ name = "rerank";        hybrid = "false"; reranker = $reranker },
    @{ name = "hybrid";        hybrid = "true";  reranker = "" },
    @{ name = "hybrid+rerank"; hybrid = "true";  reranker = $reranker }
)

function Stop-Backend {
    try {
        Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
            ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    } catch {}
    for ($i = 0; $i -lt 30; $i++) {
        if (-not (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)) { return }
        Start-Sleep -Milliseconds 500
    }
    Write-Host "  WARNING: port 8000 still listening after 15s" -ForegroundColor Yellow
}

function Wait-Ready {
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $r = Invoke-WebRequest "http://127.0.0.1:8000/healthz" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { return $true }
        } catch {}
        Start-Sleep -Seconds 1
    }
    return $false
}

$runNames = @()
foreach ($m in $models) {
    foreach ($cfg in $configs) {
        $runName = "$($cfg.name)-$($m.tag)"
        $runNames += $runName
        $summaryPath = Join-Path $citeRoot "runs\$runName\summary.json"
        if ((Test-Path $summaryPath) -and -not $Force) {
            Write-Host "`n--- ${runName}: already complete, skipping (use -Force to redo) ---" -ForegroundColor DarkGray
            continue
        }
        Write-Host "`n=== $runName (model=$($m.model), hybrid=$($cfg.hybrid), reranker=$($cfg.reranker -ne '')) ===" -ForegroundColor Cyan
        Stop-Backend

        $env:PYTHONUTF8 = "1"
        $env:LLM_PROVIDER = "ollama"
        $env:OLLAMA_MODEL = $m.model
        $env:HYBRID_RETRIEVAL = $cfg.hybrid
        $env:RERANKER_MODEL = $cfg.reranker

        $logOut = Join-Path $citeRoot "runs\.last-uvicorn.out"
        $logErr = Join-Path $citeRoot "runs\.last-uvicorn.err"
        New-Item -ItemType Directory -Force (Join-Path $citeRoot "runs") | Out-Null

        $proc = Start-Process -FilePath $paperpalPy `
            -WorkingDirectory $paperpalRoot `
            -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
            -PassThru -WindowStyle Hidden -RedirectStandardOutput $logOut -RedirectStandardError $logErr

        if (-not (Wait-Ready)) {
            Write-Host "FAILED: backend not ready in 60s (see $logErr)" -ForegroundColor Red
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            continue
        }

        Set-Location $citeRoot
        & $citePy -m citeval.run_faithfulness --name $runName --nli $nli

        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 500
    }
}

Write-Host "`n=== sweep complete ===" -ForegroundColor Green
Write-Host "Build the report with:"
Write-Host "  $citePy -m citeval.report --runs $($runNames -join ' ') --baseline dense-8b"
