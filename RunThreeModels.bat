@echo off
setlocal
title SGP4, PINN and Combined Model Experiment
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Project Python environment is missing. See README.md.
    pause
    exit /b 1
)
echo Training and evaluating three models using saved real CelesTrak and ESA data.
echo Outputs are saved under model_results and displayed in the dashboard after verification.
".venv\Scripts\python.exe" "run_three_models.py" %*
set "MODEL_RUN_EXIT=%ERRORLEVEL%"
if not "%MODEL_RUN_EXIT%"=="0" echo The run failed. Read the message above and the run error log.
pause
exit /b %MODEL_RUN_EXIT%
