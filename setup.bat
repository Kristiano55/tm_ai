@echo off
cd /d "%~dp0"
echo Installing the Python packages for the Trackmania AI (CPU-only torch)...
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
echo.
echo Done. Next: start TICK with Trackmania 2020 open, then double-click start_dashboard.bat
pause
