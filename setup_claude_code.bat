@echo off
cd /d "%~dp0"
echo === 1/3 Git ===
where git >nul 2>&1 || winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
set "PATH=%PATH%;C:\Program Files\Git\cmd"
echo === 2/3 Claude Code ===
where claude >nul 2>&1 || powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://claude.ai/install.ps1 | iex"
set "PATH=%PATH%;%USERPROFILE%\.local\bin"
echo === 3/3 Link this folder to GitHub (runs and config.json are kept) ===
if not exist ".git" (
  git init -q
  git remote add origin https://github.com/Kristiano55/tm_ai.git
)
git fetch -q origin main
git reset -q origin/main
git branch -q -M main
git branch -q --set-upstream-to=origin/main main
git status -s
echo.
echo Done. Close this window, then double-click start_claude.bat
pause
