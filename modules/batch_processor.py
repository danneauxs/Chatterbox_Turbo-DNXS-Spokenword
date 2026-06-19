import argparse
import json
import sys
from pathlib import Path

# Add project root to path to allow imports
sys.path.append('.')

from modules.tts_engine import process_single_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single batch of chunks for ChatterboxTTS.")
    parser.add_argument("--params_path", type=str, required=True, help="Path to the JSON file containing processing parameters.")
    args = parser.parse_args()

    with open(args.params_path, 'r') as f:
        params = json.load(f)

    # Re-hydrate paths and other complex objects
    params['book_dir'] = Path(params['book_dir'])
    params['voice_path'] = Path(params['voice_path'])
    if params.get('specific_text_file'):
        params['specific_text_file'] = Path(params['specific_text_file'])

    # Call the new single-batch processing function
    process_single_batch(**params)
