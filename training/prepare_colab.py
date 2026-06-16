import os
import sys
import json
import shutil
import zipfile
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(
       os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DATASET_DIR = os.path.join(ROOT, "data",
                            "datasets")
COLAB_DIR   = os.path.join(ROOT, "colab")
os.makedirs(COLAB_DIR, exist_ok=True)

# ── Windows Config ────────────────────────────────────
WINDOWS = [
    {"train_end": 2018, "test_year": 2019},
    {"train_end": 2019, "test_year": 2020},
    {"train_end": 2020, "test_year": 2021},
    {"train_end": 2021, "test_year": 2022},
    {"train_end": 2022, "test_year": 2023},
    {"train_end": 2023, "test_year": 2024},
]

# ── Package Dataset for Upload ────────────────────────
def package_datasets():
    print("  Packaging datasets for upload...")

    packages = []

    for w in WINDOWS:
        train_end = w["train_end"]
        test_year = w["test_year"]

        window_name = (
            f"window_2014_{train_end}"
            f"_test_{test_year}"
        )
        window_dir = os.path.join(
            DATASET_DIR, window_name
        )

        if not os.path.exists(window_dir):
            print(
                f"  WARNING: Missing: {window_name}"
            )
            continue

        # Create zip
        zip_name = os.path.join(
            COLAB_DIR,
            f"{window_name}.zip"
        )

        with zipfile.ZipFile(
            zip_name, "w",
            zipfile.ZIP_DEFLATED
        ) as zf:
            for file in [
                "train.jsonl",
                "test.jsonl",
                "stats.json"
            ]:
                fpath = os.path.join(
                    window_dir, file
                )
                if os.path.exists(fpath):
                    zf.write(fpath, file)

        size_mb = round(
            os.path.getsize(zip_name) /
            1024 / 1024, 2
        )
        print(
            f"  OK: {window_name}.zip "
            f"({size_mb} MB)"
        )
        packages.append({
            "name"   : window_name,
            "zip"    : zip_name,
            "size_mb": size_mb
        })

    return packages

# ── Generate Colab Notebook ───────────────────────────
def generate_colab_notebook():
    print("\n  Generating Colab notebook...")

    notebook = {
        "nbformat"      : 4,
        "nbformat_minor": 0,
        "metadata"      : {
            "accelerator": "GPU",
            "colab"      : {
                "provenance": [],
                "gpuType"   : "T4"
            },
            "kernelspec" : {
                "name"        : "python3",
                "display_name": "Python 3"
            }
        },
        "cells": []
    }

    def code_cell(source):
        return {
            "cell_type"     : "code",
            "source"        : source
                              if isinstance(
                                  source, list
                              ) else [source],
            "metadata"      : {},
            "outputs"       : [],
            "execution_count": None
        }

    def markdown_cell(source):
        return {
            "cell_type": "markdown",
            "source"   : source
                         if isinstance(
                             source, list
                         ) else [source],
            "metadata" : {}
        }

    # Cell 1 - Title
    notebook["cells"].append(markdown_cell([
        "# Trading AI - Walk Forward Fine-Tuning\n",
        "## Fine-tune Qwen 2.5 1.5B on Indian Market Data\n",
        "\n",
        "Steps:\n",
        "1. Install dependencies\n",
        "2. Upload dataset zip\n",
        "3. Fine-tune with LoRA\n",
        "4. Evaluate on test set\n",
        "5. Save and download model\n"
    ]))

    # Cell 2 - GPU check
    notebook["cells"].append(code_cell([
        "# Check GPU\n",
        "import torch\n",
        "print('GPU:', torch.cuda.get_device_name(0)"
        " if torch.cuda.is_available() else 'No GPU')\n",
        "print('CUDA:', torch.version.cuda)\n",
        "print('Memory:', round("
        "torch.cuda.get_device_properties(0)"
        ".total_memory / 1e9, 1), 'GB')\n"
    ]))

    # Cell 3 - Install
    notebook["cells"].append(code_cell([
        "# Install dependencies\n",
        "!pip install -q unsloth\n",
        "!pip install -q transformers datasets\n",
        "!pip install -q trl peft accelerate\n",
        "!pip install -q bitsandbytes\n",
        "print('Dependencies installed')\n"
    ]))

    # Cell 4 - Upload dataset
    notebook["cells"].append(code_cell([
        "# Upload your dataset zip file\n",
        "from google.colab import files\n",
        "import zipfile\n",
        "import os\n",
        "\n",
        "print('Please upload your window zip file')\n",
        "uploaded = files.upload()\n",
        "\n",
        "# Extract zip\n",
        "for fname in uploaded.keys():\n",
        "    print(f'Extracting {fname}...')\n",
        "    with zipfile.ZipFile(fname, 'r') as z:\n",
        "        z.extractall('/content/dataset')\n",
        "\n",
        "# List files\n",
        "print(os.listdir('/content/dataset'))\n"
    ]))

    # Cell 5 - Load dataset
    notebook["cells"].append(code_cell([
        "# Load training dataset\n",
        "import json\n",
        "\n",
        "def load_jsonl(path):\n",
        "    data = []\n",
        "    with open(path) as f:\n",
        "        for line in f:\n",
        "            line = line.strip()\n",
        "            if line:\n",
        "                data.append(json.loads(line))\n",
        "    return data\n",
        "\n",
        "train_data = load_jsonl('/content/dataset/train.jsonl')\n",
        "test_data  = load_jsonl('/content/dataset/test.jsonl')\n",
        "\n",
        "print(f'Train samples: {len(train_data)}')\n",
        "print(f'Test samples : {len(test_data)}')\n",
        "\n",
        "# Preview\n",
        "sample = train_data[0]\n",
        "print('Sample prompt (first 500 chars):')\n",
        "print(sample['instruction'][:500])\n",
        "print('Sample output:')\n",
        "print(sample['output'][:300])\n"
    ]))

    # Cell 6 - Load model
    notebook["cells"].append(code_cell([
        "# Load Qwen 2.5 1.5B with Unsloth\n",
        "from unsloth import FastLanguageModel\n",
        "import torch\n",
        "\n",
        "MAX_SEQ_LENGTH = 2048\n",
        "DTYPE          = None\n",
        "LOAD_IN_4BIT   = True\n",
        "\n",
        "model, tokenizer = FastLanguageModel.from_pretrained(\n",
        "    model_name     = 'unsloth/Qwen2.5-1.5B-Instruct',\n",
        "    max_seq_length = MAX_SEQ_LENGTH,\n",
        "    dtype          = DTYPE,\n",
        "    load_in_4bit   = LOAD_IN_4BIT,\n",
        ")\n",
        "\n",
        "print('Model loaded successfully')\n",
        "total = sum(p.numel() for p in model.parameters())\n",
        "print(f'Model params: {total/1e6:.0f}M')\n"
    ]))

    # Cell 7 - Add LoRA
    notebook["cells"].append(code_cell([
        "# Add LoRA adapters\n",
        "model = FastLanguageModel.get_peft_model(\n",
        "    model,\n",
        "    r              = 16,\n",
        "    target_modules = [\n",
        "        'q_proj', 'k_proj', 'v_proj', 'o_proj',\n",
        "        'gate_proj', 'up_proj', 'down_proj'\n",
        "    ],\n",
        "    lora_alpha     = 16,\n",
        "    lora_dropout   = 0,\n",
        "    bias           = 'none',\n",
        "    use_gradient_checkpointing = 'unsloth',\n",
        "    random_state   = 42,\n",
        "    use_rslora     = False,\n",
        "    loftq_config   = None,\n",
        ")\n",
        "\n",
        "trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
        "total     = sum(p.numel() for p in model.parameters())\n",
        "print(f'Trainable: {trainable/1e6:.1f}M / {total/1e6:.0f}M ({trainable/total*100:.1f}%)')\n"
    ]))

    # Cell 8 - Format dataset
    notebook["cells"].append(code_cell([
        "# Format dataset for training\n",
        "from datasets import Dataset\n",
        "\n",
        "SYSTEM_PROMPT = (\n",
        "    'You are a professional trading '\n",
        "    'intelligence system specializing '\n",
        "    'in Indian markets. Analyze market '\n",
        "    'data and provide structured '\n",
        "    'JSON trading decisions.'\n",
        ")\n",
        "\n",
        "def format_sample(sample):\n",
        "    messages = [\n",
        "        {'role': 'system',    'content': SYSTEM_PROMPT},\n",
        "        {'role': 'user',      'content': sample['instruction']},\n",
        "        {'role': 'assistant', 'content': sample['output']}\n",
        "    ]\n",
        "    return tokenizer.apply_chat_template(\n",
        "        messages,\n",
        "        tokenize              = False,\n",
        "        add_generation_prompt = False\n",
        "    )\n",
        "\n",
        "train_texts   = [format_sample(s) for s in train_data]\n",
        "train_dataset = Dataset.from_dict({'text': train_texts})\n",
        "\n",
        "print(f'Dataset size: {len(train_dataset)}')\n",
        "print('Sample (first 200 chars):')\n",
        "print(train_dataset[0]['text'][:200])\n"
    ]))

    # Cell 9 - Train
    notebook["cells"].append(code_cell([
        "# Fine-tune with SFTTrainer\n",
        "from trl import SFTTrainer\n",
        "from transformers import TrainingArguments\n",
        "from unsloth import is_bfloat16_supported\n",
        "\n",
        "trainer = SFTTrainer(\n",
        "    model              = model,\n",
        "    tokenizer          = tokenizer,\n",
        "    train_dataset      = train_dataset,\n",
        "    dataset_text_field = 'text',\n",
        "    max_seq_length     = MAX_SEQ_LENGTH,\n",
        "    dataset_num_proc   = 2,\n",
        "    args = TrainingArguments(\n",
        "        per_device_train_batch_size = 2,\n",
        "        gradient_accumulation_steps = 4,\n",
        "        warmup_steps                = 5,\n",
        "        num_train_epochs            = 3,\n",
        "        learning_rate               = 2e-4,\n",
        "        fp16        = not is_bfloat16_supported(),\n",
        "        bf16        = is_bfloat16_supported(),\n",
        "        logging_steps               = 10,\n",
        "        optim                       = 'adamw_8bit',\n",
        "        weight_decay                = 0.01,\n",
        "        lr_scheduler_type           = 'linear',\n",
        "        seed                        = 42,\n",
        "        output_dir                  = '/content/outputs',\n",
        "    ),\n",
        ")\n",
        "\n",
        "gpu_stats = torch.cuda.get_device_properties(0)\n",
        "max_memory = round(gpu_stats.total_memory / 1024**3, 1)\n",
        "print(f'GPU: {gpu_stats.name}')\n",
        "print(f'Max memory: {max_memory} GB')\n",
        "print('Starting training...')\n",
        "\n",
        "trainer_stats = trainer.train()\n",
        "\n",
        "print('Training complete!')\n",
        "print(f'Training loss: {trainer_stats.training_loss:.4f}')\n"
    ]))

    # Cell 10 - Evaluate
    notebook["cells"].append(code_cell([
        "# Evaluate on test set\n",
        "import json\n",
        "from collections import Counter\n",
        "\n",
        "FastLanguageModel.for_inference(model)\n",
        "\n",
        "def predict(instruction):\n",
        "    messages = [\n",
        "        {'role': 'system', 'content': SYSTEM_PROMPT},\n",
        "        {'role': 'user',   'content': instruction}\n",
        "    ]\n",
        "    inputs = tokenizer.apply_chat_template(\n",
        "        messages,\n",
        "        tokenize              = True,\n",
        "        add_generation_prompt = True,\n",
        "        return_tensors        = 'pt'\n",
        "    ).to('cuda')\n",
        "\n",
        "    with torch.no_grad():\n",
        "        outputs = model.generate(\n",
        "            input_ids      = inputs,\n",
        "            max_new_tokens = 512,\n",
        "            temperature    = 0.1,\n",
        "            do_sample      = True,\n",
        "        )\n",
        "\n",
        "    response = tokenizer.decode(\n",
        "        outputs[0][inputs.shape[1]:],\n",
        "        skip_special_tokens = True\n",
        "    )\n",
        "    return response\n",
        "\n",
        "# Evaluate on first 50 samples\n",
        "eval_samples = test_data[:50]\n",
        "action_match = 0\n",
        "valid_json   = 0\n",
        "results      = []\n",
        "\n",
        "print('Evaluating on 50 test samples...')\n",
        "\n",
        "for i, sample in enumerate(eval_samples):\n",
        "    try:\n",
        "        response = predict(sample['instruction'])\n",
        "\n",
        "        if '```' in response:\n",
        "            response = response.split('```')[1]\n",
        "            if response.startswith('json'):\n",
        "                response = response[4:]\n",
        "\n",
        "        pred   = json.loads(response)\n",
        "        actual = json.loads(sample['output'])\n",
        "        valid_json += 1\n",
        "\n",
        "        if pred.get('action') == actual.get('action'):\n",
        "            action_match += 1\n",
        "\n",
        "        results.append({\n",
        "            'predicted': pred.get('action'),\n",
        "            'actual'   : actual.get('action'),\n",
        "            'match'    : pred.get('action') == actual.get('action')\n",
        "        })\n",
        "\n",
        "        if (i + 1) % 10 == 0:\n",
        "            print(f'  Processed {i+1}/50...')\n",
        "\n",
        "    except Exception as e:\n",
        "        results.append({'error': str(e)})\n",
        "\n",
        "print(f'Valid JSON   : {valid_json}/50')\n",
        "print(f'Action Match : {action_match}/50 ({action_match/50*100:.0f}%)')\n",
        "\n",
        "pred_actions = Counter(r.get('predicted') for r in results if r.get('predicted'))\n",
        "print(f'Predicted Actions: {dict(pred_actions)}')\n"
    ]))

    # Cell 11 - Save model
    notebook["cells"].append(code_cell([
        "# Save fine-tuned model\n",
        "import os\n",
        "\n",
        "SAVE_PATH = '/content/trading_ai_model'\n",
        "\n",
        "model.save_pretrained(SAVE_PATH)\n",
        "tokenizer.save_pretrained(SAVE_PATH)\n",
        "\n",
        "print(f'Model saved to {SAVE_PATH}')\n",
        "for f in os.listdir(SAVE_PATH):\n",
        "    size = os.path.getsize(os.path.join(SAVE_PATH, f)) / 1024**2\n",
        "    print(f'  {f}: {size:.1f} MB')\n"
    ]))

    # Cell 12 - Save GGUF
    notebook["cells"].append(code_cell([
        "# Save as GGUF for Ollama\n",
        "GGUF_PATH = '/content/trading_ai_gguf'\n",
        "\n",
        "model.save_pretrained_gguf(\n",
        "    GGUF_PATH,\n",
        "    tokenizer,\n",
        "    quantization_method = 'q4_k_m'\n",
        ")\n",
        "\n",
        "print('GGUF model saved!')\n",
        "for f in os.listdir(GGUF_PATH):\n",
        "    size = os.path.getsize(os.path.join(GGUF_PATH, f)) / 1024**2\n",
        "    print(f'  {f}: {size:.1f} MB')\n"
    ]))

    # Cell 13 - Download
    notebook["cells"].append(code_cell([
        "# Download model to your PC\n",
        "import shutil\n",
        "from google.colab import files\n",
        "\n",
        "# Zip LoRA weights\n",
        "print('Zipping LoRA weights...')\n",
        "shutil.make_archive(\n",
        "    '/content/trading_ai_lora',\n",
        "    'zip',\n",
        "    '/content/trading_ai_model'\n",
        ")\n",
        "\n",
        "# Download LoRA\n",
        "print('Downloading LoRA weights...')\n",
        "files.download('/content/trading_ai_lora.zip')\n",
        "\n",
        "# Download GGUF\n",
        "import os\n",
        "gguf_files = [\n",
        "    f for f in os.listdir('/content/trading_ai_gguf')\n",
        "    if f.endswith('.gguf')\n",
        "]\n",
        "for f in gguf_files:\n",
        "    print(f'Downloading {f}...')\n",
        "    files.download(f'/content/trading_ai_gguf/{f}')\n",
        "\n",
        "print('Download complete!')\n"
    ]))

    # Save notebook
    nb_path = os.path.join(
        COLAB_DIR,
        "trading_ai_finetune.ipynb"
    )
    with open(nb_path, "w",
              encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"  OK: Notebook saved -> {nb_path}")
    return nb_path

# ── Generate Instructions ─────────────────────────────
def generate_instructions(packages):
    print("\n  Generating instructions...")

    total_size = sum(
        p["size_mb"] for p in packages
    )

    lines = [
        "TRADING AI - GOOGLE COLAB FINE-TUNING GUIDE",
        "=" * 60,
        "",
        "WHAT YOU HAVE BUILT",
        "-" * 60,
        "  6 Walk-Forward Training Windows",
        "  Total Train Samples : ~29,000",
        "  Total Test Samples  : ~77,000",
        f"  Total Dataset Size  : {total_size:.1f} MB",
        "",
        "FILES TO UPLOAD TO COLAB",
        "-" * 60,
        f"  Location: {COLAB_DIR}",
        "",
        "  Datasets (upload ONE at a time):",
    ]

    for p in packages:
        wname = p["name"].replace(
            "window_2014_", ""
        ).replace("_test_", " -> Test ")
        lines.append(
            f"  - {os.path.basename(p['zip'])}"
            f"  ({p['size_mb']} MB)"
            f"  [{wname}]"
        )

    lines += [
        "",
        "  Notebook:",
        "  - trading_ai_finetune.ipynb",
        "",
        "STEP BY STEP GUIDE",
        "-" * 60,
        "",
        "STEP 1 - Open Google Colab",
        "  -> Go to: https://colab.research.google.com",
        "  -> Sign in with Google account (free)",
        "",
        "STEP 2 - Upload Notebook",
        "  -> File -> Upload notebook",
        "  -> Select: trading_ai_finetune.ipynb",
        "",
        "STEP 3 - Enable GPU",
        "  -> Runtime -> Change runtime type",
        "  -> Hardware accelerator -> T4 GPU",
        "  -> Click Save",
        "",
        "STEP 4 - Run Cells 1-3 (Install)",
        "  -> Run GPU check cell",
        "  -> Run install cell (takes 2-3 mins)",
        "",
        "STEP 5 - Upload Dataset",
        "  -> Run the upload cell",
        "  -> Click Choose Files",
        "  -> Upload: window_2014_2018_test_2019.zip",
        "    (Start with this - smallest dataset)",
        "",
        "STEP 6 - Run All Remaining Cells",
        "  -> Run cells in order: 4 -> 5 -> 6 -> 7",
        "     -> 8 -> 9 (training starts here)",
        "  -> Training takes ~20-40 mins on T4",
        "",
        "STEP 7 - Evaluate",
        "  -> Run evaluation cell (cell 10)",
        "  -> Check action match accuracy",
        "",
        "STEP 8 - Download Model",
        "  -> Run download cell (cell 13)",
        "  -> Two files download:",
        "    a) trading_ai_lora.zip (LoRA weights)",
        "    b) model.gguf (for Ollama)",
        "",
        "STEP 9 - Deploy Locally",
        "  -> Extract trading_ai_lora.zip",
        "  -> Place .gguf file in models folder",
        "  -> Run deploy.bat (Windows)",
        "",
        "TRAINING ORDER (RECOMMENDED)",
        "-" * 60,
        "  Run windows in this order:",
        "",
        "  Window 1: 2014-2018 -> Test 2019  (baseline)",
        "  Window 2: 2014-2019 -> Test 2020  (includes COVID)",
        "  Window 3: 2014-2020 -> Test 2021  (recovery)",
        "  Window 4: 2014-2021 -> Test 2022  (bear market)",
        "  Window 5: 2014-2022 -> Test 2023  (recovery)",
        "  Window 6: 2014-2023 -> Test 2024  (latest data)",
        "",
        "  BEST MODEL = Window 6 (most recent data)",
        "",
        "EXPECTED RESULTS",
        "-" * 60,
        "  Valid JSON rate    : >90%",
        "  Action accuracy    : 55-70%",
        "  Training time      : 20-40 mins per window",
        "  Model size (LoRA)  : ~50 MB",
        "  Model size (GGUF)  : ~900 MB",
        "",
        "AFTER DOWNLOADING",
        "-" * 60,
        "  1. Save .gguf to: trading-ai/models/",
        "",
        "  2. Create Ollama Modelfile:",
        "     FROM ./trading_ai_model.gguf",
        "     SYSTEM 'You are a professional trading",
        "     intelligence system for Indian markets.'",
        "",
        "  3. Register with Ollama:",
        "     ollama create trading-ai -f Modelfile",
        "",
        "  4. Update intelligence.py:",
        "     Change: model = 'qwen2.5:1.5b'",
        "     To    : model = 'trading-ai'",
        "",
        "  5. Run pipeline:",
        "     python pipelines/intelligence.py",
        "",
        "=" * 60,
        "  Your fine-tuned Trading AI will be dramatically",
        "  better at Indian market decisions",
        "=" * 60,
    ]

    instructions = "\n".join(lines)

    # Save instructions
    inst_path = os.path.join(
        COLAB_DIR, "INSTRUCTIONS.txt"
    )
    with open(inst_path, "w",
              encoding="utf-8") as f:
        f.write(instructions)

    print(instructions)
    print(f"\n  OK: Instructions saved -> {inst_path}")
    return inst_path

# ── Generate Ollama Modelfile ─────────────────────────
def generate_modelfile():
    modelfile = (
        "FROM ./trading_ai_model.gguf\n"
        "\n"
        "SYSTEM \"\"\"You are a professional trading"
        " intelligence\n"
        "system specializing in Indian stock markets.\n"
        "\n"
        "You analyze market data including:\n"
        "- Price action and OHLCV data\n"
        "- Technical indicators (RSI, MACD, EMA, ADX)\n"
        "- Volatility metrics (ATR, HV, IV proxy)\n"
        "- Market regime and trend strength\n"
        "- Calendar and seasonal patterns\n"
        "\n"
        "You provide structured JSON trading decisions"
        " with:\n"
        "- Market condition assessment\n"
        "- Recommended action (buy/sell/hold/avoid)\n"
        "- Trading strategy selection\n"
        "- Confidence level\n"
        "- Risk assessment\n"
        "- Clear reasoning\n"
        "\n"
        "You are trained on 10 years of Indian market"
        " data\n"
        "from 2014 to 2024 across NIFTY, BANKNIFTY"
        " and\n"
        "top 50 NSE stocks.\n"
        "\n"
        "Always respond with valid JSON only.\"\"\"\n"
        "\n"
        "PARAMETER temperature 0.1\n"
        "PARAMETER top_p 0.9\n"
        "PARAMETER num_predict 512\n"
    )

    mf_path = os.path.join(
        COLAB_DIR, "Modelfile"
    )
    with open(mf_path, "w",
              encoding="utf-8") as f:
        f.write(modelfile)

    print(f"  OK: Modelfile saved -> {mf_path}")
    return mf_path

# ── Generate Deploy Script ────────────────────────────
def generate_deploy_script():
    script = (
        "#!/bin/bash\n"
        "# Trading AI - Local Deployment Script\n"
        "# Run this after downloading from Colab\n"
        "\n"
        "echo '==================================='\n"
        "echo '  Trading AI - Deploy Model'\n"
        "echo '==================================='\n"
        "\n"
        "# Check Ollama\n"
        "echo ''\n"
        "echo 'Step 1: Checking Ollama...'\n"
        "ollama --version\n"
        "if [ $? -ne 0 ]; then\n"
        "    echo 'ERROR: Ollama not installed'\n"
        "    exit 1\n"
        "fi\n"
        "\n"
        "# Find GGUF file\n"
        "MODEL_DIR='./models'\n"
        "GGUF_FILE=$(ls $MODEL_DIR/*.gguf"
        " 2>/dev/null | head -1)\n"
        "\n"
        "if [ -z '$GGUF_FILE' ]; then\n"
        "    echo 'ERROR: No .gguf file in"
        " models folder'\n"
        "    exit 1\n"
        "fi\n"
        "\n"
        "echo 'Found model: $GGUF_FILE'\n"
        "\n"
        "# Create Ollama model\n"
        "echo ''\n"
        "echo 'Step 2: Creating Ollama model...'\n"
        "cp colab/Modelfile $MODEL_DIR/Modelfile\n"
        "cd $MODEL_DIR\n"
        "ollama create trading-ai -f Modelfile\n"
        "cd ..\n"
        "\n"
        "# Test model\n"
        "echo ''\n"
        "echo 'Step 3: Testing model...'\n"
        "ollama run trading-ai"
        " 'Return JSON: {action: hold}'\n"
        "\n"
        "echo ''\n"
        "echo '==================================='\n"
        "echo '  Trading AI model deployed!'\n"
        "echo '  Update intelligence.py:'\n"
        "echo '  model = trading-ai'\n"
        "echo '==================================='\n"
    )

    script_path = os.path.join(
        COLAB_DIR, "deploy.sh"
    )
    with open(script_path, "w",
              encoding="utf-8") as f:
        f.write(script)

    bat = (
        "@echo off\n"
        "REM Trading AI - Deploy Fine-tuned Model\n"
        "REM Run after downloading from Colab\n"
        "\n"
        "echo ===================================\n"
        "echo   Trading AI - Deploy Model\n"
        "echo ===================================\n"
        "\n"
        "REM Check Ollama\n"
        "ollama --version\n"
        "if errorlevel 1 (\n"
        "    echo ERROR: Ollama not installed\n"
        "    exit /b 1\n"
        ")\n"
        "\n"
        "REM Find GGUF file\n"
        "echo.\n"
        "echo Looking for GGUF model file...\n"
        "for %%f in (models\\*.gguf) do (\n"
        "    set GGUF=%%f\n"
        "    echo Found: %%f\n"
        ")\n"
        "\n"
        "if not defined GGUF (\n"
        "    echo ERROR: No .gguf in models folder\n"
        "    echo Download from Colab first\n"
        "    exit /b 1\n"
        ")\n"
        "\n"
        "REM Create Ollama model\n"
        "echo.\n"
        "echo Creating Ollama model...\n"
        "copy colab\\Modelfile models\\Modelfile\n"
        "cd models\n"
        "ollama create trading-ai -f Modelfile\n"
        "cd ..\n"
        "\n"
        "REM Test\n"
        "echo.\n"
        "echo Testing model...\n"
        "ollama run trading-ai"
        " \"Return JSON: {action: hold}\"\n"
        "\n"
        "echo.\n"
        "echo ===================================\n"
        "echo   Trading AI model deployed!\n"
        "echo   Update intelligence.py:\n"
        "echo   model = trading-ai\n"
        "echo ===================================\n"
        "pause\n"
    )

    bat_path = os.path.join(
        COLAB_DIR, "deploy.bat"
    )
    with open(bat_path, "w",
              encoding="utf-8") as f:
        f.write(bat)

    print(
        f"  OK: Deploy scripts saved -> {COLAB_DIR}"
    )
    return script_path

# ── Main ──────────────────────────────────────────────
def prepare_all():
    print("=" * 60)
    print("  Trading AI - Colab Preparation")
    print(
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 60)

    # Package datasets
    packages = package_datasets()

    # Generate notebook
    nb_path = generate_colab_notebook()

    # Generate Modelfile
    mf_path = generate_modelfile()

    # Generate deploy scripts
    generate_deploy_script()

    # Generate instructions
    inst_path = generate_instructions(packages)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  COLAB PREPARATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Files created in: {COLAB_DIR}")
    print(f"\n  - trading_ai_finetune.ipynb")
    print(f"  - Modelfile")
    print(f"  - deploy.bat  (Windows)")
    print(f"  - deploy.sh   (Linux/Mac)")
    print(f"  - INSTRUCTIONS.txt")

    for p in packages:
        print(
            f"  - "
            f"{os.path.basename(p['zip'])}"
            f"  ({p['size_mb']} MB)"
        )

    print(f"\n  NEXT STEPS:")
    print(f"  1. Open Google Colab")
    print(f"  2. Upload trading_ai_finetune.ipynb")
    print(f"  3. Enable T4 GPU")
    print(f"  4. Upload first dataset zip")
    print(f"  5. Run all cells")
    print(f"  6. Download trained model")
    print(f"  7. Run deploy.bat to install locally")

if __name__ == "__main__":
    prepare_all()