@echo off
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo  MAS vs Single LLM — Demo Setup
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

:: Move to repo root (one level up from demo\)
cd /d "%~dp0.."

:: Create virtual environment
echo.
echo [1/3] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: python not found. Install Python 3.11+ from https://python.org/downloads
    pause
    exit /b 1
)

:: Activate and install
echo [2/3] Installing dependencies (this may take a few minutes)...
call venv\Scripts\activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
if errorlevel 1 (
    echo ERROR: pip install failed. Check your internet connection.
    pause
    exit /b 1
)

:: Copy .env template
echo [3/3] Creating demo\.env from template...
if not exist "demo\.env" (
    copy "demo\.env.example" "demo\.env" >nul
    echo  Created demo\.env — please fill in your OLLAMA_BASE_URL and JUPYTERHUB_TOKEN
) else (
    echo  demo\.env already exists — skipping copy
)

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo  Setup complete!
echo.
echo  Next steps:
echo    1. Open demo\.env in a text editor
echo    2. Fill in OLLAMA_BASE_URL and JUPYTERHUB_TOKEN
echo    3. Run the demo:
echo         venv\Scripts\activate
echo         python demo\run_demo.py
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pause
