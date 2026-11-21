@echo off
cd /d "%~dp0"
set "PATH=%PATH%;%USERPROFILE%\.local\bin;C:\Program Files\Git\cmd"
git pull -q
claude
