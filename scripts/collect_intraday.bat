@echo off
REM Daily intraday-bar collector wrapper (run by Windows Task Scheduler).
REM Appends the day's 5m/15m bars to data\intraday\ so history grows past
REM yfinance's 60-day wall. Safe to run repeatedly (idempotent dedupe).

set "PROJECT=C:\Users\KetanMohite\OneDrive - INTELLINUM\Desktop\git\allGit\trading-ai"
set "PYTHON=C:\Users\KetanMohite\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJECT%"
if not exist "%PROJECT%\logs" mkdir "%PROJECT%\logs"

REM Reliable date stamp (locale-independent) via PowerShell.
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "STAMP=%%i"
"%PYTHON%" training\collect_intraday.py >> "%PROJECT%\logs\intraday_collect_%STAMP%.log" 2>&1
