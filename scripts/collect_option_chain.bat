@echo off
REM Live option-chain snapshot (run by Task Scheduler every 5 min, market hrs).
REM Appends one ML-ready row + strike-level snapshot per call to
REM data\option_chain\. Safe to run repeatedly (de-duped by timestamp).

set "PROJECT=C:\Users\KetanMohite\OneDrive - INTELLINUM\Desktop\git\allGit\trading-ai"
set "PYTHON=C:\Users\KetanMohite\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJECT%"
if not exist "%PROJECT%\logs" mkdir "%PROJECT%\logs"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "STAMP=%%i"
"%PYTHON%" training\collect_option_chain.py >> "%PROJECT%\logs\option_chain_%STAMP%.log" 2>&1
