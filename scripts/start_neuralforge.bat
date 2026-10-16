@echo off
setlocal enabledelayedexpansion
title NeuralForge - AI Research Studio
color 0b

echo =====================================================================
echo          NEURALFORGE - AI RESEARCH STUDIO  [by hexstereo]
echo =====================================================================
echo.

cd /d "%~dp0\.."

:: Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    pause
    exit /b 1
)

echo [INFO] Starting NeuralForge Desktop Application...
start "" ".venv\Scripts\pythonw.exe" "desktop_app.py"

echo [SUCCESS] NeuralForge Desktop Application launched!
timeout /t 3 /nobreak >nul
exit /b 0
