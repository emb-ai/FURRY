param(
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$Python = 'python',
    [string]$Model = 'models/yolo11s-pose_openvino_model',
    [string]$Serial = '',
    [string]$TagLayout = '',
    [double]$Seconds = 60,
    [int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $workspace
try {
    $loggerArgs = @('-m', 'inference_logger', 'record', '--output', $Output,
                    '--model', $Model, '--seconds', $Seconds, '--port', $Port)
    if ($Serial) { $loggerArgs += @('--serial', $Serial) }
    if ($TagLayout) { $loggerArgs += @('--tag-layout', $TagLayout) }
    & $Python @loggerArgs
    if ($LASTEXITCODE -ne 0) { throw "Inference logger failed with code $LASTEXITCODE" }
} finally { Pop-Location }
