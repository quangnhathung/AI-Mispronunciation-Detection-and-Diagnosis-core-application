@echo off
REM ===========================================================================
REM  Retrain CNN-BiLSTM-CTC model on L2Arctic dataset
REM  Dataset: D:\HungQuang_WorkSpace\MDD_AI\data\raw (24 speakers, ~27K utterances)
REM  Config:  configs\config.yaml
REM ===========================================================================

cd /d "%~dp0"

REM ── Activate Conda/virtual env (adjust path to match your setup) ──
REM call conda activate mdd_env
REM or if using venv:
REM call D:\python_env\internal\envs\mdd_env\Scripts\activate.bat

echo.
echo ===========================================================================
echo  Starting CNN-BiLSTM-CTC Training
echo  Dataset: D:\HungQuang_WorkSpace\MDD_AI\data\raw
echo  Config:  configs\config.yaml
echo  Epochs:  150
echo  Device:  auto (cuda if available, else cpu)
echo ===========================================================================
echo.

python train.py ^
    --config configs\config.yaml ^
    --data_dir "D:\HungQuang_WorkSpace\MDD_AI\data\raw" ^
    --epochs 150 ^
    --device cuda

echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Training completed.
) else (
    echo [ERROR] Training failed with code %ERRORLEVEL%.
)
pause
