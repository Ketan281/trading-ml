@echo off
REM Trading AI - Deploy Fine-tuned Model
REM Run after downloading from Colab

echo ===================================
echo   Trading AI - Deploy Model
echo ===================================

REM Check Ollama
ollama --version
if errorlevel 1 (
    echo ERROR: Ollama not installed
    exit /b 1
)

REM Find GGUF file
echo.
echo Looking for GGUF model file...
for %%f in (models\*.gguf) do (
    set GGUF=%%f
    echo Found: %%f
)

if not defined GGUF (
    echo ERROR: No .gguf in models folder
    echo Download from Colab first
    exit /b 1
)

REM Create Ollama model
echo.
echo Creating Ollama model...
copy colab\Modelfile models\Modelfile
cd models
ollama create trading-ai -f Modelfile
cd ..

REM Test
echo.
echo Testing model...
ollama run trading-ai "Return JSON: {action: hold}"

echo.
echo ===================================
echo   Trading AI model deployed!
echo   Update intelligence.py:
echo   model = trading-ai
echo ===================================
pause
