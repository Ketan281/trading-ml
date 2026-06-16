@echo off
REM API pre-compute — refresh cached dashboards/book/screen so /query is instant.
REM Schedule every 5 min during market hours (weekdays).

set "PROJECT=C:\Users\KetanMohite\OneDrive - INTELLINUM\Desktop\git\allGit\trading-ai"
set "PYTHON=C:\Users\KetanMohite\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe"
set "PYTHONIOENCODING=utf-8"

cd /d "%PROJECT%"
if not exist "%PROJECT%\logs" mkdir "%PROJECT%\logs"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "STAMP=%%i"
"%PYTHON%" api\precompute.py >> "%PROJECT%\logs\precompute_%STAMP%.log" 2>&1
