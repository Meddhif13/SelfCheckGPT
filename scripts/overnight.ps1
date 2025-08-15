# Requires: PowerShell 5+ on Windows
param(
    [string]$OpenAIModel = 'gpt-4o-mini',
    [switch]$SkipOpenAI,
    [string]$ResultsDir = 'results',
    [int]$LimitSmoke = 20,
    [int]$LimitFull = 100
)

$ErrorActionPreference = 'Stop'
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

function Write-Section([string]$Title) {
    Write-Host "`n==== $Title ====" -ForegroundColor Cyan
}

function Resolve-Python {
    $venvPy = Join-Path $PSScriptRoot '.\.venv\Scripts\python.exe'
    if (Test-Path $venvPy) { return $venvPy }
    return 'python'
}

function Invoke-Step {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [scriptblock]$ScriptBlock,
        [int]$Retries = 2,
        [int]$DelaySec = 10,
        [switch]$Fatal
    )
    Write-Section $Name
    $success = $false
    $attempt = 0
    while ($true) {
        try {
            & $ScriptBlock
            if ($LASTEXITCODE -ne 0) { throw "ExitCode $LASTEXITCODE" }
            Write-Host "[OK] $Name" -ForegroundColor Green
        $success = $true
            break
        } catch {
            $attempt++
            Write-Warning "[FAIL $attempt/$Retries] ${Name}: $($_.Exception.Message)"
            if ($attempt -gt $Retries) {
                if ($Fatal) { throw }
                else { break }
            }
            Start-Sleep -Seconds $DelaySec
        }
    }
    return $success
}

# Timestamped transcript
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$repoRoot = Split-Path -Parent $PSScriptRoot
$resultsRoot = Join-Path $repoRoot $ResultsDir
$overnightDir = Join-Path $resultsRoot 'overnight'
New-Item -ItemType Directory -Force -Path $overnightDir | Out-Null
$runDir = Join-Path $overnightDir $ts
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$TranscriptPath = Join-Path $runDir "transcript_${ts}.log"
Start-Transcript -LiteralPath $TranscriptPath | Out-Null

try {
    # Env setup (HF offline/cache, do not echo secrets)
    $env:HF_HOME = (Resolve-Path (Join-Path $repoRoot 'hf-cache')).Path
    if (-not $env:TRANSFORMERS_OFFLINE) { $env:TRANSFORMERS_OFFLINE = '1' }
    # Load OpenAI key silently if file exists and not set
    $openAIKeyPath = Join-Path $PSScriptRoot '.secrets\openai.key'
    if (-not [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
        Write-Host "OpenAI key present: Yes (from environment)"
    } elseif (Test-Path $openAIKeyPath) {
        $env:OPENAI_API_KEY = (Get-Content -LiteralPath $openAIKeyPath -Raw).Trim()
        Write-Host "OpenAI key present: Yes (from .secrets)"
    } else {
        Write-Host "OpenAI key present: No" -ForegroundColor Yellow
    }

    # Helpers for structure and docs
    function New-RunPath([string]$Relative) { Join-Path $runDir $Relative }
    function Write-Json($obj, [string]$path) { $obj | ConvertTo-Json -Depth 8 | Out-File -FilePath $path -Encoding utf8 }
    $manifest = New-Object System.Collections.ArrayList

    # Tooling summary
    $python = Resolve-Python
    Write-Section 'Environment summary'
    Write-Host "OS: $(Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty Caption)"
    Write-Host "PSVersion: $($PSVersionTable.PSVersion)"
    $pyInfo = @'
import sys
print('Python:', sys.version.replace('\n',' '))
try:
    import openai
    print('openai:', getattr(openai, '__version__', 'n/a'))
except Exception as e:
    print('openai: n/a', e)
try:
    import transformers
    print('transformers:', transformers.__version__)
except Exception as e:
    print('transformers: n/a', e)
try:
    import torch
    print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
except Exception as e:
    print('torch: n/a', e)
'@
    $envDir = New-RunPath 'env'
    New-Item -ItemType Directory -Force -Path $envDir | Out-Null
    & $python -c $pyInfo | Tee-Object -FilePath (Join-Path $envDir 'python.txt') | Out-Null
    try { & nvidia-smi.exe | Tee-Object -FilePath (Join-Path $envDir 'gpu.txt') | Out-Null } catch { 'nvidia-smi not found (OK if CPU)' | Out-File -FilePath (Join-Path $envDir 'gpu.txt') -Encoding utf8 }

    # Save run config
    $runConfig = [ordered]@{
        timestamp = $ts
        openaiModel = $OpenAIModel
        skipOpenAI = [bool]$SkipOpenAI
        limits = @{ smoke = $LimitSmoke; full = $LimitFull }
        resultsRoot = $resultsRoot
        runDir = $runDir
        hfHome = $env:HF_HOME
        transformersOffline = $env:TRANSFORMERS_OFFLINE
    }
    Write-Json $runConfig (Join-Path $runDir 'run_config.json')

    # Paths
    $bertscoreModel = Join-Path $repoRoot 'hf-cache\roberta-large'
    $promptHFModel = Join-Path $repoRoot 'hf-cache\lmqg__flan-t5-base-squad-qg'

    # Preflight: OpenAI connectivity (optional)
    $OpenAIOK = $false
    if (-not $SkipOpenAI) {
        $pingStart = Get-Date
        $ok = Invoke-Step -Name "OpenAI ping ($OpenAIModel)" -Retries 1 -ScriptBlock {
            & $python (Join-Path $PSScriptRoot 'check_openai.py') --model $OpenAIModel
        }
        $pingEnd = Get-Date
        if ($ok) { $OpenAIOK = $true } else { Write-Warning 'OpenAI check failed; will skip OpenAI experiments.' }
        [void]$manifest.Add([ordered]@{ name='openai_ping'; ok=$ok; start=$pingStart; end=$pingEnd; model=$OpenAIModel })
    } else {
        Write-Host 'Skipping OpenAI experiments by request.'
    }

    $modelTag = (($OpenAIModel -replace '[^a-zA-Z0-9_-]','').ToLower())

    # 1) Prompt metric (OpenAI)
    if ($OpenAIOK) {
        $outDir1 = New-RunPath (Join-Path 'online' ("prompt_smoke_" + $modelTag))
        Invoke-Step -Name 'OpenAI prompt smoke' -Retries 2 -DelaySec 15 -ScriptBlock {
            & $python (Join-Path $repoRoot 'run_experiments.py') `
                --metrics prompt `
                --limit $LimitSmoke `
                --llm-model $OpenAIModel `
                --output-dir $outDir1 `
                --deterministic `
                --verbose
        } | Out-Null
        [void]$manifest.Add([ordered]@{ name='openai_prompt_smoke'; ok=($LASTEXITCODE -eq 0); outDir=$outDir1; limit=$LimitSmoke; model=$OpenAIModel })
    }

    # 2) N-gram resample (OpenAI)
    if ($OpenAIOK) {
        $outDir2 = New-RunPath (Join-Path 'online' ("resample_smoke_" + $modelTag))
        Invoke-Step -Name 'OpenAI resample smoke' -Retries 2 -DelaySec 15 -ScriptBlock {
            & $python (Join-Path $repoRoot 'run_experiments.py') `
                --metrics ngram `
                --limit $LimitSmoke `
                --resample `
                --sample-count 3 `
                --llm-model $OpenAIModel `
                --temperature 0.7 `
                --top-p 0.9 `
                --top-k 50 `
                --output-dir $outDir2 `
                --deterministic `
                --verbose
        } | Out-Null
        [void]$manifest.Add([ordered]@{ name='openai_resample_smoke'; ok=($LASTEXITCODE -eq 0); outDir=$outDir2; limit=$LimitSmoke; model=$OpenAIModel })
    }

    # 3) Combined run (OpenAI + local models)
    if ($OpenAIOK) {
        $outDir3 = New-RunPath (Join-Path 'online' ("combined_" + $LimitFull + '_' + $modelTag))
        Invoke-Step -Name 'OpenAI combined 100' -Retries 1 -ScriptBlock {
            & $python (Join-Path $repoRoot 'run_experiments.py') `
                --metrics ngram nli bertscore prompt `
                --limit $LimitFull `
                --sample-count 10 `
                --resample `
                --llm-model $OpenAIModel `
                --nli-batch-size 16 `
                --nli-max-length 160 `
                --bertscore-model $bertscoreModel `
                --output-dir $outDir3 `
                --deterministic `
                --verbose
        } | Out-Null
        [void]$manifest.Add([ordered]@{ name='openai_combined'; ok=($LASTEXITCODE -eq 0); outDir=$outDir3; limit=$LimitFull; model=$OpenAIModel })
    }

    # 4) GPU-tuned offline HF demo
    $outDir4 = New-RunPath (Join-Path 'offline' 'gpu_demo_smoke_tuned')
    Invoke-Step -Name 'GPU demo smoke (HF offline, tuned thresholds)' -Retries 1 -ScriptBlock {
        & $python (Join-Path $repoRoot 'run_experiments.py') `
            --metrics ngram nli bertscore prompt `
            --limit $LimitSmoke `
            --output-dir $outDir4 `
            --deterministic `
            --nli-batch-size 16 `
            --nli-max-length 160 `
            --bertscore-model $bertscoreModel `
            --prompt-backend hf `
            --prompt-hf-model $promptHFModel `
            --prompt-hf-task text2text-generation `
            --prompt-hf-device cuda `
            --prompt-hf-max-new-tokens 24 `
            --tune-thresholds `
            --tune-split train `
            --verbose
    } | Out-Null
    [void]$manifest.Add([ordered]@{ name='offline_gpu_demo_smoke_tuned'; ok=($LASTEXITCODE -eq 0); outDir=$outDir4; limit=$LimitSmoke })

    # Persist manifest and write a brief README
    Write-Json $manifest (Join-Path $runDir 'manifest.json')
    $readme = @()
    $readme += "# SelfCheckGPT overnight run"
    $readme += "- Timestamp: $ts"
    $readme += "- Model: $OpenAIModel"
    $readme += "- Transformers offline: $($env:TRANSFORMERS_OFFLINE)"
    $readme += "- HF cache: $($env:HF_HOME)"
    $readme += ""
    $readme += "## Outputs"
    foreach ($m in $manifest) {
        # Robust boolean extraction
        $okVal = $false
        try {
            if ($m.ok -is [bool]) { $okVal = $m.ok }
            elseif ($m.ok -is [System.Array] -and $m.ok.Length -gt 0 -and ($m.ok[-1] -is [bool])) { $okVal = $m.ok[-1] }
        } catch {}
        $status = if ($okVal) { 'OK' } else { 'FAIL' }

        # Compute relative path only if outDir exists
        $rel = $null
        if ($m.PSObject.Properties.Name -contains 'outDir' -and $m.outDir) {
            try {
                $resolvedOut = Resolve-Path -LiteralPath $m.outDir -ErrorAction Stop
                $rel = $resolvedOut.Path.Substring((Resolve-Path $runDir).Path.Length).TrimStart('\\')
            } catch {}
        }

        if ($rel) { $readme += "- [$status] $($m.name): .\\$rel" }
        else { $readme += "- [$status] $($m.name)" }
    }
    $readme += ""
    $readme += "See transcript: .\\$(Split-Path -Leaf $TranscriptPath)"
    $readme -join "`r`n" | Out-File -FilePath (Join-Path $runDir 'README.md') -Encoding utf8

    Write-Section 'Done'
    Write-Host "Transcript: $TranscriptPath"
    Write-Host "Artifacts in: $((Resolve-Path $runDir).Path)"
} finally {
    try { Stop-Transcript | Out-Null } catch {}
}
