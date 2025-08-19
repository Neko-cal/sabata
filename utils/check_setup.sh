#!/bin/bash
### Copyright 2025 RobotsMali AI4D Lab.
### (Retains original license)

# Function to check if a manifest file exists and is not empty
check_manifest() {
  file_path="$1"
  if [ -s "$file_path" ]; then
    count=$(wc -l < "$file_path")
    echo "✔ Manifest '$file_path' found with $count entries."
    return 0
  else
    echo "❌ ERROR: Manifest file '$file_path' is missing or empty."
    echo "         Please run 'python prepare_sabian_dataset.py' first."
    return 1
  fi
}

# Function to check if the tokenizer directory contains the required files
check_tokenizer() {
  tokenizer_dir="$1"
  required_files=("tokenizer.model" "tokenizer.vocab" "vocab.txt")
  missing_files=()

  if [ ! -d "$tokenizer_dir" ]; then
      echo "❌ ERROR: Tokenizer directory not found at $tokenizer_dir"
      echo "         Please run 'bash create_sabian_tokenizer.sh' first."
      return 1
  fi

  for file in "${required_files[@]}"; do
    if [ ! -f "$tokenizer_dir/$file" ]; then
      missing_files+=("$file")
    fi
  done

  if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✔ Tokenizer is ready. All required files are present in $tokenizer_dir"
    return 0
  else
    echo "❌ ERROR: The following tokenizer files are missing in $tokenizer_dir: ${missing_files[*]}"
    return 1
  fi
}

# Function to check if dependencies are installed
check_dependencies() {
  system_dependencies=("pip" "ffmpeg" "libsndfile1" "python3")
  python_packages=("pydub" "nemo" "megatron" "wandb")

  missing_deps=()

  # Check system dependencies
  for dep in "${system_dependencies[@]}"; do
    if ! command -v "$dep" &>/dev/null && ! dpkg -l | grep -q "$dep"; then
      missing_deps+=("$dep (system)")
    fi
  done

  # Check Python dependencies
  for pkg in "${python_packages[@]}"; do
    if ! python3 -c "import $pkg" &>/dev/null; then
      missing_deps+=("$pkg (Python)")
    fi
  done

  # Report results
  if [ ${#missing_deps[@]} -eq 0 ]; then
    echo "✔ All dependencies are installed."
    return 0
  else
    echo "❌ ERROR: The following dependencies are missing:"
    printf "%s\n" "${missing_deps[@]}"
    echo "Run the installation script to resolve this issue."
    return 1
  fi
}

# Function to check wandb authentication
check_wandb() {
  if [ -z "$WANDB_API_KEY" ]; then
    echo "❌ ERROR: No WANDB_API_KEY found in environment variables."
    echo "Please set it using: export WANDB_API_KEY=your_api_key"
    return 1
  fi
  
  wandb login --relogin "$WANDB_API_KEY"
  if [ $? -eq 0 ]; then
    echo "✔ Weights & Biases authentication successful."
    return 0
  else
    echo "❌ ERROR: Weights & Biases login failed. Ensure your API key is correct."
    return 1
  fi
}
# --- Main Check Logic ---
echo "--- Running Pre-flight Checks for Sabian ASR Fine-tuning ---"

# Define paths based on the new workflow
DATASET_DIR="./sabian_dataset"
TOKENIZER_DIR_BASE="./sabian-tokenizer"
TOKENIZER_DIR_FULL="$TOKENIZER_DIR_BASE/tokenizer_spe_bpe_v1024" # Update vocab size if you changed it

# Check for manifests
check_manifest "$DATASET_DIR/manifests/train_manifest.json" || exit 1
check_manifest "$DATASET_DIR/manifests/validation_manifest.json" || exit 1
check_manifest "$DATASET_DIR/manifests/test_manifest.json" || exit 1

# Check for tokenizer
check_tokenizer "$TOKENIZER_DIR_FULL" || exit 1

# Check for dependencies and wandb login (optional, uncomment if needed)
# check_dependencies || exit 1
# check_wandb || exit 1

echo -e "\n✔ All checks passed. System is ready for training."
