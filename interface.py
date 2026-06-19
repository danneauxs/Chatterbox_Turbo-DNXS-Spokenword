"""
==============================================================================
ENHANCED GENTTS AUDIOBOOK GENERATOR - MODULAR VERSION
==============================================================================
A high-performance, enterprise-grade TTS audiobook production system built on
ChatterboxTTS with advanced quality control, memory management, and performance
optimization features.

This is the main orchestration module that coordinates all the modular components:
- Text processing (modules/text_processor.py)
- Audio processing (modules/audio_processor.py) 
- TTS engine management (modules/tts_engine.py)
- File operations (modules/file_manager.py)
- Progress tracking (modules/progress_tracker.py)
- Resume functionality (modules/resume_handler.py)

USAGE MODES:
1. Interactive book selection with voice and parameter configuration
2. Batch processing queue for multiple books
3. Combine-only mode for re-assembling existing chunks
4. Single chunk testing for parameter optimization

AUTHOR: Enhanced by Claude (Anthropic) for optimized audiobook production
VERSION: 3.0 Modular - Clean Architecture Edition
LICENSE: Open source - Use responsibly for legal content only
==============================================================================
"""

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="LlamaModel is using LlamaSdpaAttention")
warnings.filterwarnings("ignore", message="We detected that you are passing `past_key_values`")
warnings.filterwarnings("ignore")

# Import core libraries
import os
import sys
import signal
import torch
import argparse
from pathlib import Path
from src.chatterbox.tts_turbo import ChatterboxTurboTTS

# Set environment and suppress warnings
sys.stdout.flush()
# Cache setup is handled in config.config - don't override here

# Import modular components
from config.config import *
from modules.text_processor import (
    sentence_chunk_text, smart_punctuate, detect_content_boundaries
)
from src.chatterbox.tts_turbo import punc_norm
from modules.audio_processor import (
    smart_audio_validation, add_contextual_silence, pause_for_chunk_review
)
from modules.tts_engine import (
    monitor_gpu_activity, optimize_memory_usage, load_optimized_model,
    patch_alignment_layer, process_one_chunk, process_book_folder, get_best_available_device
)
from modules.file_manager import (
    list_voice_samples, ensure_voice_sample_compatibility, chunk_sort_key,
    convert_to_m4b, add_metadata_to_m4b, combine_audio_chunks, find_book_files,
    save_chunk_info
)
from modules.progress_tracker import log_chunk_progress, log_console, log_run, setup_logging
from modules.resume_handler import process_book_folder_resume, resume_book_from_chunk

from tools.combine_only import run_combine_only_mode

# Optional CLI override for ASR threshold
ASR_THRESHOLD_OVERRIDE = None

# ============================================================================
# GLOBAL SHUTDOWN HANDLING
# ============================================================================

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    print(f"\n⚠️ {YELLOW}Shutdown requested. Finishing current chunk...{RESET}")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

# ============================================================================
# BOOK SELECTION AND PARAMETER PROMPTS
# ============================================================================

def prompt_book_selection(book_dirs, already_selected):
    """Interactive book selection from available directories"""
    available = [d for d in book_dirs if d not in already_selected]
    if not available:
        print("No more books available.")
        return None
    
    print("\nAvailable books:")
    for i, book_dir in enumerate(available, 1):
        print(f" [{i}] {book_dir.name}")
    
    while True:
        try:
            choice = input("Select book number: ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    return available[idx]
            print("Invalid selection. Try again.")
        except (ValueError, KeyboardInterrupt):
            return None

def prompt_voice_selection(voice_files):
    """Interactive voice selection from available samples"""
    print("\nAvailable voices:")
    for i, voice_file in enumerate(voice_files, 1):
        print(f" [{i}] {voice_file.stem}")
    
    while True:
        try:
            choice = input("Select voice number: ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(voice_files):
                    return voice_files[idx]
            print("Invalid selection. Try again.")
        except (ValueError, KeyboardInterrupt):
            return None

def prompt_tts_params():
    """Interactive TTS parameter configuration"""
    print("\nTTS Parameters:")
    
    def get_float_input(prompt, default):
        """Prompt user for a floating-point number input, using an optional default value. Returns the entered float or the default if no valid input is provided.
        Args:
        prompt (str): The question to ask the user.
        default (float): The default value to use if no input is provided.
        Returns:
        float: The user-entered float value or the specified default.
        """
        while True:
            try:
                value = input(f"{prompt} [{default}]: ").strip() or str(default)
                return float(value)
            except ValueError:
                print(f"❌ Invalid input. Please enter a valid number.")
    
    def get_yes_no_input(prompt, default=True):
        """Prompt user to input a Yes/No response.
        Args:
        prompt (str): The question prompt.
        default (bool): Default value if user presses Enter.
        Returns:
        bool: True for 'y' or 'yes', False for 'n' or 'no'.
        ---
        Prompt user to select from a list of choices.
        Args:
        prompt (str): The question prompt.
        choices (list): List of available choices.
        default_idx (int): Index of default choice.
        Returns:
        any: Selected choice.
        """
        while True:
            default_str = "Y/n" if default else "y/N"
            value = input(f"{prompt} [{default_str}]: ").strip().lower()
            if not value:
                return default
            if value in ['y', 'yes']:
                return True
            elif value in ['n', 'no']:
                return False
            else:
                print("❌ Please enter 'y' for yes or 'n' for no.")
    
    def get_choice_input(prompt, choices, default_idx=0):
        """Prompts the user for a choice from a list of options and returns the selected choice.
        Args:
        prompt (str): The message to display to the user.
        choices (list): A list of available choices.
        default_idx (int, optional): The index of the default choice. Defaults to 0.
        Returns:
        str: The chosen item from the list.
        """
        while True:
            try:
                choice = input(f"{prompt} [{default_idx + 1}]: ").strip()
                if not choice:
                    return choices[default_idx]
                idx = int(choice) - 1
                if 0 <= idx < len(choices):
                    return choices[idx]
                print(f"❌ Please enter a number between 1 and {len(choices)}")
            except ValueError:
                print(f"❌ Please enter a valid number")
    
    # VADER sentiment analysis option
    use_vader = get_yes_no_input("🎭 Use VADER sentiment analysis to adjust TTS params per chunk?", True)
    
    if use_vader:
        print("✅ VADER enabled - TTS params will be adjusted based on chunk sentiment")
        print("   (Base values will be modified up/down per chunk)")
    else:
        print("❌ VADER disabled - TTS params will be fixed for all chunks")
        print("   (Same values used for every chunk)")
    
    # ASR validation option
    use_asr = get_yes_no_input("🎤 Enable ASR validation for quality control?", False)
    asr_config = None
    asr_threshold = DEFAULT_ASR_THRESHOLD
    
    if use_asr:
        print("\n🔍 Analyzing system capabilities...")
        
        # Import here to avoid circular imports
        from modules.system_detector import get_system_profile, recommend_asr_models, print_system_summary
        
        profile = get_system_profile()
        print_system_summary(profile)
        
        recommendations = recommend_asr_models(profile)
        
        print("\nASR Model Recommendations:")
        print("🟢 [1] SAFE:     Fast processing, basic accuracy")
        print("🟡 [2] MODERATE: Balanced speed/accuracy (recommended)")  
        print("🔴 [3] INSANE:   Best accuracy, may stress system")
        
        choice_labels = ['safe', 'moderate', 'insane']
        choice_idx = get_choice_input("Select ASR level", list(range(len(choice_labels))), 1)  # Default to moderate
        selected_level = choice_labels[choice_idx]
        
        selected_config = recommendations[selected_level]
        
        print(f"\n✅ Selected {selected_level.upper()} ASR configuration:")
        primary = selected_config['primary']
        fallback = selected_config['fallback']
        print(f"   Primary:  {primary['model']} on {primary['device'].upper()}")
        print(f"   Fallback: {fallback['model']} on {fallback['device'].upper()}")
        
        if selected_level == 'insane':
            print("⚠️  WARNING: INSANE mode may cause memory pressure and slower performance")
        
        asr_config = {
            'enabled': True,
            'level': selected_level,
            'primary_model': primary['model'],
            'primary_device': primary['device'],
            'fallback_model': fallback['model'],
            'fallback_device': fallback['device']
        }

        threshold_default = ASR_THRESHOLD_OVERRIDE if ASR_THRESHOLD_OVERRIDE is not None else DEFAULT_ASR_THRESHOLD
        threshold_input = get_float_input("ASR similarity threshold (0.50 - 1.00)", threshold_default)
        asr_threshold = max(0.5, min(1.0, threshold_input))
    else:
        print("❌ ASR disabled - no output validation will be performed")
        asr_config = {'enabled': False}
    
    print("\nBase TTS Parameters:")
    exaggeration = get_float_input("Exaggeration", DEFAULT_EXAGGERATION)
    cfg_weight = get_float_input("CFG Weight", DEFAULT_CFG_WEIGHT)
    temperature = get_float_input("Temperature", DEFAULT_TEMPERATURE)
    
    print("\nAdvanced Sampling Parameters:")
    min_p = get_float_input("Min-P (0.0 disables)", DEFAULT_MIN_P)
    top_p = get_float_input("Top-P (1.0 disables)", DEFAULT_TOP_P)
    repetition_penalty = get_float_input("Repetition Penalty", DEFAULT_REPETITION_PENALTY)
    
    return {
        'exaggeration': exaggeration,
        'cfg_weight': cfg_weight,
        'temperature': temperature,
        'min_p': min_p,
        'top_p': top_p,
        'repetition_penalty': repetition_penalty,
        'use_vader': use_vader,
        'asr_config': asr_config,
        'asr_threshold': asr_threshold
    }

# ============================================================================
# MAIN BOOK PROCESSING FUNCTIONS
# ============================================================================

# process_book_folder() now imported from modules.tts_engine

# ============================================================================
# PIPELINE AND UTILITY FUNCTIONS
# ============================================================================

def pipeline_book_processing(books_to_process):
    """
    Processes a queue of books, calling the main processing function for each.
    """
    completed_books = []
    device = get_best_available_device()
    print(f"🚀 Starting processing on device: {device}")

    for i, book_info in enumerate(books_to_process, 1):
        book_dir = book_info['book_dir']
        voice_path = book_info['voice_path']
        tts_params = book_info['tts_params']
        
        print(f"\n=====================================================================")
        print(f"▶️ PROCESSING BOOK {i}/{len(books_to_process)}: {book_dir.name}")
        print(f"=====================================================================")
        
        try:
            # Extract ASR setting from the tts_params collected from the user
            asr_config = tts_params.get('asr_config', {})
            enable_asr = asr_config.get('enabled', False)

            # Determine ASR threshold (CLI override wins)
            asr_threshold = tts_params.get('asr_threshold', DEFAULT_ASR_THRESHOLD)
            if ASR_THRESHOLD_OVERRIDE is not None:
                asr_threshold = ASR_THRESHOLD_OVERRIDE

            # Call the main processing function from tts_engine
            final_m4b_path, _, _ = process_book_folder(
                book_dir=book_dir,
                voice_path=voice_path,
                tts_params=tts_params,
                device=device,
                enable_asr=enable_asr,
                config_params={'asr_config': asr_config},  # Pass the detailed asr_config
                asr_threshold=asr_threshold
            )
            
            if final_m4b_path and final_m4b_path.exists():
                print(f"✅ SUCCESS: Completed {book_dir.name}")
                completed_books.append(book_dir.name)
            else:
                print(f"⚠️ WARNING: Processing finished for {book_dir.name}, but the final M4B file was not found.")

        except Exception as e:
            print(f"❌ FATAL ERROR processing {book_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Moving to next book in queue...")

    return completed_books

# run_combine_only_mode() now imported from tools.combine_only

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for GenTTS processing"""
    log_console("Enhanced GenTTS v3.0 Modular - Convert text to audiobook", "GREEN")
    
    # Get available books and voices
    book_dirs = [d for d in TEXT_INPUT_ROOT.iterdir() if d.is_dir()]
    voice_files = list_voice_samples()
    
    if not book_dirs:
        print(f"❌ No book directories found in {TEXT_INPUT_ROOT}")
        return
    
    if not voice_files:
        print(f"❌ No voice samples found in {VOICE_SAMPLES_DIR}")
        return
    
    # Interactive selection
    selected_books = []
    while True:
        book_dir = prompt_book_selection(book_dirs, selected_books)
        if not book_dir:
            break
        
        voice_path = prompt_voice_selection(voice_files)
        if not voice_path:
            break
        
        # Ensure voice compatibility
        voice_path = ensure_voice_sample_compatibility(voice_path)
        
        tts_params = prompt_tts_params()
        
        selected_books.append({
            'book_dir': book_dir,
            'voice_path': voice_path,
            'tts_params': tts_params
        })
        
        if input("\nAdd another book? [y/N]: ").lower() != 'y':
            break
    
    if not selected_books:
        print("No books selected.")
        return
    
    # Display configuration
    print(f"\n📋 Processing Queue:")
    for i, book_info in enumerate(selected_books, 1):
        voice_path = book_info['voice_path']
        if voice_path:
            voice_name = Path(voice_path).stem if isinstance(voice_path, str) else voice_path.stem
        else:
            voice_name = "Unknown"
        print(f"  {i}. {book_info['book_dir'].name} -> {voice_name}")
    
    print(f"  Workers: {MAX_WORKERS}")
    print(f"  VRAM Threshold: {VRAM_SAFETY_THRESHOLD}GB")
    print(f"  ASR Enabled: {ENABLE_ASR}")
    print(f"  Hum Detection: {ENABLE_HUM_DETECTION}")
    
    # Process queue
    completed_books = pipeline_book_processing(selected_books)
    
    print(f"\n{GREEN}Processing complete: {len(completed_books)}/{len(selected_books)} books{RESET}")

def main_with_resume():
    """Main entry point with resume option"""
    print(f"{RED}Enhanced ChatterboxTTS Batch Audiobook Generator\n{RESET}")
    
    print("Select an action:")
    print(" 1. Convert a book (normal processing)")
    print(" 2. Resume a book from specific chunk")
    print(" 3. Re-concatenate audio_chunks into audiobook (combine only)")
    
    mode = input("Enter option number [1/2/3]: ").strip()
    
    if mode == "2":
        start_chunk = int(input("Enter chunk number to resume from: "))
        return resume_book_from_chunk(start_chunk)
    elif mode == "3":
        return run_combine_only_mode()
    else:
        return main()

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable-asr', action='store_true', help="Enable ASR validation")
    parser.add_argument('--asr-threshold', type=float, help="Override ASR similarity threshold (0.50-1.00)")
    parser.add_argument('--resume', type=int, help="Resume processing from specific chunk number")
    args, unknown = parser.parse_known_args()
    
    # Override ASR setting if specified via command line
    if args.enable_asr:
        ENABLE_ASR = True
    
    if args.asr_threshold is not None:
        ASR_THRESHOLD_OVERRIDE = max(0.5, min(1.0, args.asr_threshold))
    
    if args.resume:
        print(f"Resuming from chunk {args.resume}")
        resume_book_from_chunk(args.resume)
    else:
        main()
