@echo off
setlocal

echo ==========================================
echo Eggcellent ID - first time setup
echo ==========================================

cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher not found.
    echo Please install Python 3.11 or newer first.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment...
    py -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call ".venv\Scripts\activate.bat"

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo Setup finished successfully.
echo Next steps:
echo 1. Double-click run_app.bat to open the Streamlit app
echo 2. If you want phone camera access from another device, double-click start_ngrok.bat
echo.
pause
