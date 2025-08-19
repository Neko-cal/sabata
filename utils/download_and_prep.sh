#!/bin/bash

# --- Configuration ---
# Paths are now relative to the root directory
DATASET_PREP_SCRIPT="utils/prepare_sabian_dataset.py"
TOKENIZER_CREATION_SCRIPT="utils/process_asr_text_tokenizer.py"

# Output directories
TOKENIZER_DIR="./sabian-tokenizer"
MANIFEST_DIR="./sabian-asr/manifests"

# Parameters
VOCAB_SIZE=1024
TOKENIZER_TYPE="bpe"

# Function to display help message
display_help() {
  echo "Usage: $0 [OPTIONS]"
  echo "\nThis script prepares the Sabian dataset and creates a tokenizer."
  echo "\nOptions:"
  echo "  --vocab_size=<size>     Vocabulary size for tokenizer (default: 1024)"
  echo "  --spe_type=<type>       Type of SentencePiece tokenizer (default: bpe)"
  echo "  -h, --help              Show this help message and exit"
  exit 0
}

# Parse arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --vocab_size=*) VOCAB_SIZE="${1#*=}" ;;
    --spe_type=*) TOKENIZER_TYPE="${1#*=}" ;;
    -h|--help) display_help ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

# --- Step 1: Prepare the Sabian Dataset and Manifests ---
echo "--> Running dataset preparation script..."
python3 "$DATASET_PREP_SCRIPT"
if [ $? -ne 0 ]; then
  echo "❌ ERROR: Dataset preparation failed."
  exit 1
fi
echo "✔ Dataset and manifests prepared successfully in ./sabian-asr/"

# --- Step 2: Create the Tokenizer ---
echo "--> Running the tokenizer creation script..."
python3 "$TOKENIZER_CREATION_SCRIPT" \
  --manifest="$MANIFEST_DIR/train-manifest.json" \
  --vocab_size="$VOCAB_SIZE" \
  --data_root="$TOKENIZER_DIR" \
  --tokenizer="spe" \
  --spe_type="$TOKENIZER_TYPE" \
  --spe_character_coverage=1.0

if [ $? -ne 0 ]; then
  echo "❌ ERROR: Tokenizer creation failed."
  exit 1
fi
echo "✔ Tokenizer created successfully in $TOKENIZER_DIR"
echo "\n✅ All preparation steps are complete."
