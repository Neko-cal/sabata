"""
This script downloads the 'mutisya/sabian' dataset from Hugging Face,
and prepares it for NeMo ASR training. It creates:
1. Manifest files (train, validation, test) in .json format.
2. A single text file containing all training transcriptions for tokenizer creation.
"""
import os
import json
import librosa
from datasets import load_dataset, Audio
from tqdm import tqdm
import warnings

# Suppress warnings from librosa about audioread
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

def create_manifest(dataset, manifest_path):
    """Creates a NeMo-compatible manifest file from a Hugging Face dataset split."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    
    with open(manifest_path, 'w', encoding='utf-8') as fout:
        for item in tqdm(dataset, desc=f"Processing {os.path.basename(manifest_path)}"):
            try:
                # The 'audio' feature in Hugging Face datasets provides the path
                audio_path = item['audio']['path']
                
                # NeMo requires absolute paths in manifests
                abs_audio_path = os.path.abspath(audio_path)
                
                if not os.path.exists(abs_audio_path):
                    print(f"Warning: Audio file not found at {abs_audio_path}. Skipping.")
                    continue

                # Get audio duration using librosa
                duration = librosa.get_duration(path=abs_audio_path)
                
                # Get the transcript
                text = item['sentence']

                # Create a JSON entry for the manifest
                entry = {
                    'audio_filepath': abs_audio_path,
                    'duration': duration,
                    'text': text
                }
                fout.write(json.dumps(entry) + '\n')
            except Exception as e:
                print(f"Error processing item: {item}. Error: {e}")


def prepare_text_for_tokenizer(dataset, text_path):
    """Extracts all transcripts into a single text file for tokenizer training."""
    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    with open(text_path, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset, desc="Writing text corpus for tokenizer"):
            f.write(item['sentence'] + '\n')

if __name__ == "__main__":
    # --- Configuration ---
    dataset_name = "mutisya/sabian"
    output_dir = "sabian_dataset"
    sample_rate = 16000
    
    # --- Main Logic ---
    print(f"Loading '{dataset_name}' dataset from Hugging Face...")
    # This will download and cache the dataset. 
    # We cast the audio feature to ensure it's loaded at the correct sample rate.
    sabian_dataset = load_dataset(dataset_name)
    sabian_dataset = sabian_dataset.cast_column("audio", Audio(sampling_rate=sample_rate))


    # Create directories for manifests and tokenizer data
    manifests_dir = os.path.join(output_dir, "manifests")
    tokenizer_data_dir = os.path.join(output_dir, "tokenizer_data")
    
    # Create manifests for each split
    create_manifest(sabian_dataset['train'], os.path.join(manifests_dir, 'train_manifest.json'))
    create_manifest(sabian_dataset['validation'], os.path.join(manifests_dir, 'validation_manifest.json'))
    create_manifest(sabian_dataset['test'], os.path.join(manifests_dir, 'test_manifest.json'))
    
    # Prepare a single text file from the training data to train the tokenizer
    all_text_path = os.path.join(tokenizer_data_dir, 'all_text.txt')
    prepare_text_for_tokenizer(sabian_dataset['train'], all_text_path)
    
    print("\n--- Summary ---")
    print(f"Manifests created in: {manifests_dir}")
    print(f"  - train_manifest.json")
    print(f"  - validation_manifest.json")
    print(f"  - test_manifest.json")
    print(f"Text corpus for tokenizer created at: {all_text_path}")
    print("\nDataset preparation complete. You can now create the tokenizer.")
