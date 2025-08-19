#!/bin/bash

# Set default values for sabian dataset
DATASET_PREP_SCRIPT="prepare_sabian_dataset.py"
TOKENIZER_DIR="./sabian-tokenizer"
MANIFEST_DIR="./sabian-asr/manifests"
VOCAB_SIZE=1024 # Or whatever size you prefer
TOKENIZER_TYPE="bpe" # As requested

# Function to display help message
display_help() {
  echo "Usage: $0 [OPTIONS]"
  echo "\nOptions:"
  echo "  --vocab_size=<size>           Vocabulary size for tokenizer (default: 1024)"
  echo "  --spe_type=<type>             Type of SentencePiece tokenizer (default: bpe)"
  echo "  --tokenizer_dir=<path>        Directory for tokenizer output (default: ./sabian-tokenizer)"
  echo "  --create_tokenizer_script_path=<path> Path to NeMo's tokenizer script"
  echo "  -h, --help                    Show this help message and exit"
  exit 0
}

# Parse arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --vocab_size=*) vocab_size="${1#*=}" ;;
    --spe_type=*) tokenizer_type="${1#*=}" ;;
    --tokenizer_dir=*) TOKENIZER_DIR="${1#*=}" ;;
    --create_tokenizer_script_path=*) create_tokenizer_script_path="${1#*=}" ;;
    -h|--help) display_help ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

# --- Step 1: Prepare the Sabian Dataset and Manifests ---
echo "Running dataset preparation script for Sabian..."
python3 $DATASET_PREP_SCRIPT
if [ $? -ne 0 ]; then
  echo "❌ ERROR: Dataset preparation failed."
  exit 1
fi
echo "✔ Dataset and manifests prepared successfully."

# --- Step 2: Create the Tokenizer ---
# Check if the tokenizer script path is provided
if [ -z "$create_tokenizer_script_path" ]; then
    echo "❌ ERROR: --create_tokenizer_script_path is required."
    display_help
fi

echo "Running the tokenizer script..."
python3 $create_tokenizer_script_path \
  --manifest="$MANIFEST_DIR/train-manifest.json" \
  --vocab_size="$vocab_size" \
  --data_root="$TOKENIZER_DIR" \
  --tokenizer="spe" \
  --spe_type="$tokenizer_type" \
  --spe_character_coverage=1.0 # Use 1.0 for most languages

if [ $? -ne 0 ]; then
  echo "❌ ERROR: Tokenizer creation failed."
  exit 1
fi
echo "✔ Tokenizer created successfully in $TOKENIZER_DIR"
