@echo off
cd /d "%~dp0"
python tm_ai.py check "C:\Users\Kris\Documents\Trackmania\Maps\Better Training - 14.Map.Gbx" > check_output.txt 2>&1
type check_output.txt
pause
