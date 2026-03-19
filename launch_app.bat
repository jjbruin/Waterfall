@echo off
title Waterfall XIRR
cd /d "%~dp0"

echo Starting Waterfall XIRR...
echo.

REM Start Flask API in background
start "Flask API" /min cmd /c ".venv\Scripts\python -m flask_app.run"

REM Wait for Flask to start
timeout /t 3 /nobreak >nul

REM Start Vue dev server in background
start "Vue Frontend" /min cmd /c "cd vue_app && npm run dev"

REM Wait for Vue to start
timeout /t 5 /nobreak >nul

REM Open browser
start http://localhost:5173

echo Waterfall XIRR is running!
echo   Flask API:    http://localhost:5000
echo   Vue Frontend: http://localhost:5173
echo.
echo Close this window to keep servers running.
echo To stop, close the "Flask API" and "Vue Frontend" windows.
pause
