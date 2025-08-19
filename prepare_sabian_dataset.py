import os
import json
from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf

def create_manifest(dataset, manifest_path, split_name):
    """Creates a NeMo-compatible manifest file from a Hugging Face dataset split."""
    print(f"Creating manifest for {split_name} split...")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset, desc=f"Processing {split_name}"):
            audio_path = item['audio']['path']
            text = item['sentence'] # IMPORTANT: Check the column name for transcripts in your dataset
            
            # Check if audio file exists and get duration
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found at {audio_path}. Skipping.")
                continue
                
            try:
                audio_info = sf.info(audio_path)
                duration = audio_info.duration
            except Exception as e:
                print(f"Could not read duration for {audio_path}. Error: {e}. Skipping.")
                continue

            # Ensure text is clean (e.g., remove special characters if needed)
            # This is a good place for language-specific text normalization
            text = text.strip()

            entry = {
                'audio_filepath': os.path.abspath(audio_path),
                'duration': duration,
                'text': text
            }
            f.write(json.dumps(entry) + '\n')
    print(f"Manifest created at: {manifest_path}")

def main():
    # --- Configuration ---
    dataset_name = "your-hf-username/sabian" # IMPORTANT: Replace with your actual dataset name
    output_dir = "sabian-asr"
    manifest_dir = os.path.join(output_dir, "manifests")

    # --- Main Logic ---
    print(f"Loading '{dataset_name}' dataset from Hugging Face...")
    # This will download and cache the dataset. The audio is stored locally.
    # trust_remote_code=True may be needed for some datasets.
    dataset = load_dataset(dataset_name, trust_remote_code=True) 
    
    os.makedirs(manifest_dir, exist_ok=True)
    
    # Check for available splits and create manifests
    # IMPORTANT: Adjust split names ('train', 'validation', 'test') to match your dataset
    if 'train' in dataset:
        create_manifest(dataset['train'], os.path.join(manifest_dir, 'train-manifest.json'), 'train')
    if 'validation' in dataset:
        create_manifest(dataset['validation'], os.path.join(manifest_dir, 'valid-manifest.json'), 'validation')
    if 'test' in dataset:
        create_manifest(dataset['test'], os.path.join(manifest_dir, 'test-manifest.json'), 'test')
        
    print("\nDataset preparation complete.")
    print(f"Manifests are located in: {manifest_dir}")

if __name__ == "__main__":
    main()
