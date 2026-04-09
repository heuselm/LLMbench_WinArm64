<#
.SYNOPSIS
    Benchmark LM Studio models for reasoning ability and generation speed.

.DESCRIPTION
    Auto-discovers every LLM/VLM currently in LM Studio via its
    /api/v0/models endpoint, writes models_config.json, lets you pick
    models interactively (or by substring filter), and for each
    (model x context-length) combination:

      1. Unloads any resident models (verified via API poll)
      2. Loads the target model via `lms.exe` with the requested ctx
      3. Verifies: target loaded, exactly 1 model in memory, actual ctx
         length matches requested (fails fast if RAM/VRAM was short)
      4. Runs speed_test.py (tok/s + TTFT, 3 reasoning prompts)
      5. Runs run_evals.py (ARC / HellaSwag / WinoGrande / GSM8K)
      6. Saves per-run JSON under results\<model>\ctx_<N>\

    After all runs, regenerates report.txt, results\all_results.json,
    results\all_results.tsv and results\overview_plot.png.

    A Windows sleep guard (SetThreadExecutionState) is installed in a
    try/finally so the PC won't sleep mid-run and will always recover
    normal sleep behaviour on exit.

.PARAMETER Model
    Substring filter against model name / lms_key / api_id. Empty string
    (the default) opens an interactive numbered picker.

.PARAMETER Size
    Suite size. Determines items per task:
      small  = 20 MC / 8 GSM8K   (~10-30 min/model, +-11% error)
      medium = 50 MC / 20 GSM8K  (+-7% error) -- default
      large  = 200 MC / 50 GSM8K (+-3.5% error, multi-hour)

.PARAMETER ContextLengths
    Comma-separated list of context lengths to test, e.g. "32768,65536".
    Each model is benchmarked once per context length.

.PARAMETER Tasks
    Comma-separated reasoning tasks to run. Use a subset (e.g. "gsm8k")
    to re-run a single task; run_evals.py merges with the existing
    eval_results.json so previously-completed task scores are preserved.

.PARAMETER SpeedOnly
    Skip the reasoning eval stage; only measure speed.

.PARAMETER EvalOnly
    Skip the speed test stage; only run reasoning tasks.

.EXAMPLE
    .\run_benchmark.ps1
    Interactive picker, medium suite, both 32K and 64K context (default).

.EXAMPLE
    .\run_benchmark.ps1 -Size small
    Quick smoke test across all models, default context lengths.

.EXAMPLE
    .\run_benchmark.ps1 -Model "Gemma" -Size small -ContextLengths "32768"
    Benchmark every model whose name contains "Gemma", small suite,
    32K context only.

.EXAMPLE
    .\run_benchmark.ps1 -Model "Qwen3.5" -Tasks "gsm8k"
    Re-run just the GSM8K task for Qwen3.5 (e.g. after fixing the
    answer extractor). ARC/HellaSwag/WinoGrande scores from the
    previous run are preserved.

.EXAMPLE
    .\run_benchmark.ps1 -SpeedOnly
    Speed-only sweep -- handy for A/B testing hardware or quant levels.

.NOTES
    Prerequisites:
      1. Run .\setup.ps1 once to create the venv.
      2. Open LM Studio -> Developer -> Local Server -> Start (port 1234).

    If blocked by execution policy:
      Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

    Tested with LM Studio 0.4.9+1 on Windows 11 ARM64 (PowerShell 5.1).

.LINK
    https://github.com/heuselm/LLMbench_WinArm64
#>

param(
    [switch]$SpeedOnly,
    [switch]$EvalOnly,
    [string]$Model = "",
    [ValidateSet("small","medium","large")]
    [string]$Size = "medium",
    [string]$ContextLengths = "32768,65536",
    [string]$Tasks = "arc_challenge,hellaswag,winogrande,gsm8k"
)

$ErrorActionPreference = "Continue"
$ScriptDir  = $PSScriptRoot
$BaseUrl    = "http://localhost:1234/v1"
$ApiV0      = "http://localhost:1234/api/v0"
$ResultsDir = "$ScriptDir\results"
$ConfigFile = "$ScriptDir\models_config.json"
$PY         = "$ScriptDir\venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    Write-Warning "venv not found -- using system Python. Run .\setup.ps1 first."
    $PY = "python"
}

$CtxList = $ContextLengths -split ',' | ForEach-Object { [int]$_.Trim() }

# --- Sleep guard (prevents Windows sleep during benchmark) ------------------
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class SleepGuard {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint esFlags);
    public const uint ES_CONTINUOUS      = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;
}
"@
function Enable-StayAwake  {
    [SleepGuard]::SetThreadExecutionState([SleepGuard]::ES_CONTINUOUS -bor [SleepGuard]::ES_SYSTEM_REQUIRED) | Out-Null
    Write-Host "  [Sleep guard ON]  System will not sleep during benchmark."
}
function Disable-StayAwake {
    [SleepGuard]::SetThreadExecutionState([SleepGuard]::ES_CONTINUOUS) | Out-Null
    Write-Host "  [Sleep guard OFF] Normal sleep behaviour restored."
}

# --- Model name generator ---------------------------------------------------
function Get-CleanModelName([string]$Id, [string]$Quantization) {
    # Strip publisher prefix (e.g. "openai/gpt-oss-20b" -> "gpt-oss-20b")
    $base = $Id -replace '^[^/]+/', ''
    # Title-case each hyphen-separated segment
    $name = ($base -split '-' | ForEach-Object {
        if ($_.Length -gt 0) { $_.Substring(0,1).ToUpper() + $_.Substring(1) } else { $_ }
    }) -join '-'
    # Append quantization if not already present
    if ($Quantization -and ($name -notmatch [regex]::Escape($Quantization))) {
        $name = "$name-$Quantization"
    }
    return $name
}

# --- Auto-discover models from LM Studio and refresh models_config.json ----
function Sync-ModelConfig {
    Write-Host "  Discovering models from LM Studio API..."
    try {
        $apiModels = @((Invoke-RestMethod -Uri "$ApiV0/models" -TimeoutSec 5 -ErrorAction Stop).data |
            Where-Object { $_.type -in @("llm", "vlm") })
    } catch {
        Write-Warning "Could not reach $ApiV0/models: $_"
        return $null
    }

    # Load existing config to preserve machine metadata and existing names
    $existing = @{}
    if (Test-Path $ConfigFile) {
        $prev = Get-Content $ConfigFile -Raw | ConvertFrom-Json
        foreach ($m in $prev.models) {
            $existing[$m.api_id] = $m
        }
        $machine  = $prev.machine
        $platform = $prev.platform
        $lmsCli   = $prev.lms_cli
    } else {
        $machine  = $env:COMPUTERNAME
        $platform = "Windows 11 ARM64"
        $lmsCli   = "$env:USERPROFILE\.lmstudio\bin\lms.exe"
    }

    # Build updated model list -- preserve existing names for known models
    $models = @()
    foreach ($m in $apiModels) {
        if ($existing.ContainsKey($m.id)) {
            # Keep existing entry (preserves name, description, etc.)
            $models += $existing[$m.id]
        } else {
            # New model -- auto-generate entry
            $cleanName = Get-CleanModelName -Id $m.id -Quantization $m.quantization
            $models += [PSCustomObject]@{
                name        = $cleanName
                lms_key     = $m.id
                api_id      = $m.id
                description = "$($m.publisher) $($m.arch) $($m.quantization) [$($m.type.ToUpper())]"
                type        = $m.type
            }
        }
    }

    # Write updated config
    $config = [PSCustomObject]@{
        machine          = $machine
        platform         = $platform
        date             = (Get-Date -Format "yyyy-MM-dd")
        lmstudio_base_url = $BaseUrl
        lms_cli          = $lmsCli
        models           = $models
    }
    $config | ConvertTo-Json -Depth 5 | Set-Content $ConfigFile -Encoding UTF8

    Write-Host "  Found $($models.Count) models -- models_config.json updated."
    return $config
}

# --- Interactive model picker ------------------------------------------------
function Select-ModelsInteractively([array]$modelList) {
    Write-Host ""
    Write-Host "  Available models:"
    Write-Host "  ---------------------------------------------------------------"
    for ($i = 0; $i -lt $modelList.Count; $i++) {
        $m = $modelList[$i]
        Write-Host ("  [{0}] {1}" -f ($i + 1), $m.name)
        Write-Host ("       {0}" -f $m.description)
    }
    Write-Host "  ---------------------------------------------------------------"
    Write-Host "  Enter numbers to benchmark (e.g. '1 3'), 'all', or ENTER for all:"
    $answer = Read-Host "  Selection"

    if ([string]::IsNullOrWhiteSpace($answer) -or $answer.Trim().ToLower() -eq 'all') {
        Write-Host "  -> All $($modelList.Count) models selected."
        return $modelList
    }

    $indices = $answer.Trim() -split '\s+' | ForEach-Object {
        $n = 0
        if ([int]::TryParse($_, [ref]$n) -and $n -ge 1 -and $n -le $modelList.Count) { $n - 1 }
        else { Write-Warning "Ignored invalid selection: '$_'" }
    } | Where-Object { $_ -ne $null }

    [array]$selected = $indices | ForEach-Object { $modelList[$_] }
    Write-Host "  -> Selected: $(($selected | ForEach-Object { $_.name }) -join ', ')"
    return $selected
}

# --- Server / model state helpers -------------------------------------------
function Test-Server {
    try {
        $null = Invoke-RestMethod -Uri "$BaseUrl/models" -TimeoutSec 5 -ErrorAction Stop
        return $true
    } catch { return $false }
}

function Get-LoadedModels {
    try {
        $resp = Invoke-RestMethod -Uri "$ApiV0/models" -TimeoutSec 5 -ErrorAction Stop
        [array]$found = $resp.data | Where-Object { $_.state -eq "loaded" }
        return $found
    } catch { return [array]@() }
}

function Assert-NothingLoaded([int]$TimeoutSec = 60) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        [array]$loaded = Get-LoadedModels
        if ($loaded.Count -eq 0) { return $true }
        $names = ($loaded | ForEach-Object { $_.id }) -join ", "
        Write-Host "  Waiting for unload... still loaded: $names" -NoNewline
        Start-Sleep -Seconds 2
        Write-Host "`r" -NoNewline
    }
    [array]$still = Get-LoadedModels
    Write-Warning "UNLOAD CHECK FAILED after ${TimeoutSec}s: $(($still | ForEach-Object { $_.id }) -join ', ')"
    return $false
}

function Invoke-UnloadAll {
    if (-not (Test-Path $LMS)) { return }
    Write-Host "  Unloading all models..."
    $out = & $LMS unload --all 2>&1
    $out | ForEach-Object { Write-Host "    $_" }
    $clean = Assert-NothingLoaded -TimeoutSec 30
    if (-not $clean) { throw "Could not verify clean unload -- aborting to prevent bad results" }
    Write-Host "  Verified: no models loaded."
}

function Invoke-LoadModel([string]$LmsKey, [string]$ApiId, [int]$CtxLen) {
    if (Test-Path $LMS) {
        Write-Host "  Loading: $LmsKey  (ctx=$CtxLen)"
        $out = & $LMS load $LmsKey --yes --context-length $CtxLen 2>&1
        $out | ForEach-Object { Write-Host "    $_" }

        # Check 1: target model reaches loaded state
        $deadline = (Get-Date).AddSeconds(180)
        $modelLoaded = $false
        while ((Get-Date) -lt $deadline) {
            [array]$loaded = Get-LoadedModels
            $target = $loaded | Where-Object { $_.id -eq $ApiId }
            if ($target) { $modelLoaded = $true; break }
            Write-Host "  Waiting for load..." -NoNewline
            Start-Sleep -Seconds 3
            Write-Host "`r" -NoNewline
        }
        if (-not $modelLoaded) {
            Write-Warning "LOAD CHECK FAILED: $ApiId did not reach 'loaded' within 180s"
            return $false
        }

        # Check 2: exactly one model in memory
        [array]$loaded = Get-LoadedModels
        if ($loaded.Count -ne 1) {
            $names = ($loaded | ForEach-Object { $_.id }) -join ", "
            Write-Warning "ISOLATION CHECK FAILED: expected 1 model, got $($loaded.Count) -- $names"
            return $false
        }

        # Check 3: context length matches
        $actualCtx = [int]$loaded[0].loaded_context_length
        if ($actualCtx -ne $CtxLen) {
            Write-Warning "CONTEXT CHECK FAILED: requested ctx=$CtxLen, got ctx=$actualCtx"
            Write-Warning "Likely cause: insufficient RAM/VRAM. Ensure clean unload before loading."
            return $false
        }

        Write-Host "  [OK] Loaded: $ApiId | ctx=$actualCtx | isolated: 1 model in memory"
        return $true
    } else {
        Write-Host "  Load manually in LM Studio: $LmsKey (ctx=$CtxLen)"
        Read-Host "  Press ENTER when ready"
        return $true
    }
}

# --- Speed test and eval ----------------------------------------------------
function Invoke-SpeedTest([string]$ModelName, [string]$ApiId, [string]$OutDir) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    Write-Host ""
    Write-Host "  [SPEED TEST] $ModelName"
    & $PY "$ScriptDir\speed_test.py" --model $ApiId --base-url $BaseUrl --output "$OutDir\speed.json"
}

function Invoke-ReasoningEval([string]$ModelName, [string]$ApiId, [string]$OutDir) {
    $evalDir = "$OutDir\eval"
    New-Item -ItemType Directory -Force -Path $evalDir | Out-Null
    Write-Host ""
    Write-Host "  [REASONING EVAL] $ModelName  (size: $Size)"
    $startTime = Get-Date
    & $PY "$ScriptDir\run_evals.py" `
        --model $ApiId --base-url $BaseUrl --output $evalDir `
        --tasks $Tasks --size $Size
    $elapsed = [int]((Get-Date) - $startTime).TotalSeconds
    Write-Host "  Eval finished in ${elapsed}s"
    Set-Content -Path "$evalDir\eval_time_seconds.txt" -Value $elapsed
}

# ===========================================================================
# MAIN
# ===========================================================================

Enable-StayAwake

try {

    Write-Host "============================================================"
    Write-Host "  LM Model Benchmark Runner"
    Write-Host "  Machine  : $env:COMPUTERNAME | Windows 11 ARM64"
    Write-Host "  Suite    : $Size"
    Write-Host "  Ctx lens : $($CtxList -join ', ')"
    Write-Host "============================================================"
    Write-Host ""

    # Pre-flight: server must be up
    if (-not (Test-Server)) {
        throw "LM Studio server not reachable at $BaseUrl -- open LM Studio and start Local Server (port 1234)"
    }
    Write-Host "  Server OK at $BaseUrl"

    # Sync model list from LM Studio API
    $config = Sync-ModelConfig
    if (-not $config) { throw "Model discovery failed -- check LM Studio server" }

    $LMS = $config.lms_cli
    if (-not (Test-Path $LMS)) {
        Write-Warning "lms CLI not found at: $LMS -- will prompt for manual load"
        $LMS = $null
    }

    [array]$allModels = $config.models

    # Determine which models to run
    if ($Model) {
        # Substring match against name, lms_key, or api_id
        [array]$selectedModels = $allModels | Where-Object {
            $_.name    -match [regex]::Escape($Model) -or
            $_.lms_key -match [regex]::Escape($Model) -or
            $_.api_id  -match [regex]::Escape($Model)
        }
        if ($selectedModels.Count -eq 0) {
            throw "No model matched '$Model'. Available: $(($allModels | ForEach-Object { $_.name }) -join ', ')"
        }
        Write-Host "  Model filter '$Model' matched: $(($selectedModels | ForEach-Object { $_.name }) -join ', ')"
    } else {
        # Interactive picker
        [array]$selectedModels = Select-ModelsInteractively -modelList $allModels
        if ($selectedModels.Count -eq 0) { throw "No models selected -- exiting." }
    }

    Write-Host ""
    $totalRuns = $selectedModels.Count * $CtxList.Count
    Write-Host "  Running $($selectedModels.Count) model(s) x $($CtxList.Count) context length(s) = $totalRuns run(s)"
    Write-Host ""

    $runsDone  = 0
    $completed = 0
    $failed    = 0

    foreach ($m in $selectedModels) {
        foreach ($ctx in $CtxList) {
            $runsDone++
            $ctxTag = "ctx_$ctx"
            $outDir = "$ResultsDir\$($m.name)\$ctxTag"

            Write-Host ""
            Write-Host ("=" * 72)
            Write-Host "  Run $runsDone/$totalRuns -- $($m.name)  [ctx=$ctx]"
            Write-Host "  $($m.description)"
            Write-Host ("=" * 72)

            Invoke-UnloadAll
            $loaded = Invoke-LoadModel -LmsKey $m.lms_key -ApiId $m.api_id -CtxLen $ctx
            if (-not $loaded) {
                Write-Warning "Skipping $($m.name) ctx=$ctx -- failed to load"
                $failed++
                continue
            }

            if (-not $EvalOnly) {
                try   { Invoke-SpeedTest -ModelName $m.name -ApiId $m.api_id -OutDir $outDir }
                catch { Write-Warning "Speed test failed: $_" }
            }

            if (-not $SpeedOnly) {
                try   { Invoke-ReasoningEval -ModelName $m.name -ApiId $m.api_id -OutDir $outDir }
                catch { Write-Warning "Reasoning eval failed: $_" }
            }

            # Save run metadata
            $meta = [ordered]@{
                model_name     = $m.name
                api_id         = $m.api_id
                context_length = $ctx
                size           = $Size
                timestamp      = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
            } | ConvertTo-Json
            New-Item -ItemType Directory -Force -Path $outDir | Out-Null
            Set-Content -Path "$outDir\run_meta.json" -Value $meta

            $completed++
            Write-Host ""
            Write-Host "  Done: $($m.name) ctx=$ctx"
        }
    }

    Invoke-UnloadAll

    # Generate report and plot
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host "  GENERATING REPORT + PLOT"
    Write-Host ("=" * 72)
    & $PY "$ScriptDir\aggregate_results.py"
    & $PY "$ScriptDir\plot_results.py"

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  Benchmark complete!"
    Write-Host "  Completed : $completed  |  Failed: $failed"
    Write-Host "  Report    : $ScriptDir\report.txt"
    Write-Host "  Plot      : $ResultsDir\overview_plot.png"
    Write-Host "  Results   : $ResultsDir\"
    Write-Host "============================================================"

} finally {
    Disable-StayAwake
}
