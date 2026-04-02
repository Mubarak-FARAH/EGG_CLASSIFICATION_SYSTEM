@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment not found.
    echo Please run install_app.bat first.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"

:: Paste your Moorcheh API key after the = sign, no quotes, no spaces
set MOORCHEH_API_KEY=mq2wDrvNjz6hfG59hdAtS2528FGTlVZj7rpbELLW

echo Starting Streamlit app...
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
pause
