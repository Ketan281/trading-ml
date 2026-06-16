#!/bin/bash
# Trading AI - Local Deployment Script
# Run this after downloading from Colab

echo '==================================='
echo '  Trading AI - Deploy Model'
echo '==================================='

# Check Ollama
echo ''
echo 'Step 1: Checking Ollama...'
ollama --version
if [ $? -ne 0 ]; then
    echo 'ERROR: Ollama not installed'
    exit 1
fi

# Find GGUF file
MODEL_DIR='./models'
GGUF_FILE=$(ls $MODEL_DIR/*.gguf 2>/dev/null | head -1)

if [ -z '$GGUF_FILE' ]; then
    echo 'ERROR: No .gguf file in models folder'
    exit 1
fi

echo 'Found model: $GGUF_FILE'

# Create Ollama model
echo ''
echo 'Step 2: Creating Ollama model...'
cp colab/Modelfile $MODEL_DIR/Modelfile
cd $MODEL_DIR
ollama create trading-ai -f Modelfile
cd ..

# Test model
echo ''
echo 'Step 3: Testing model...'
ollama run trading-ai 'Return JSON: {action: hold}'

echo ''
echo '==================================='
echo '  Trading AI model deployed!'
echo '  Update intelligence.py:'
echo '  model = trading-ai'
echo '==================================='
