@echo off
setlocal

echo ==========================================
echo Eggcellent ID - ngrok tunnel
echo ==========================================

where ngrok >nul 2>nul
if errorlevel 1 (
    echo ngrok is not installed or is not in PATH.
    echo Install ngrok first, then run:
    echo ngrok config add-authtoken YOUR_TOKEN
    echo.
    pause
    exit /b 1
)

echo Starting HTTPS tunnel for Streamlit on port 8501...
ngrok http 8501

pause
