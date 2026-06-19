#!/usr/bin/env python3
"""
ASR Worker Daemon - Runs in ASR venv with persistent Whisper models.
Communicates with main process via file-based task/result queue.

This daemon loads Whisper model once per worker and reuses it for all validations,
avoiding the 2-4 second model loading overhead per chunk that subprocess approach had.
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher

# Globals for persistent model (loaded once per worker)
_model = None
_device = None
model_size = "base"  # Module-level default

def _load_whisper_model():
    """
    Load Whisper model on CPU only.

    GPU mode is disabled because ASR should not compete with TTS models for VRAM.
    During audiobook generation, TTS models (T3 + S3Gen) occupy most GPU memory.
    Running ASR on CPU uses system RAM instead, preventing OOM errors.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        logging.error(f"Failed to import faster-whisper: {e}")
        raise

    logging.info(f"🖥️  Loading Whisper {model_size} on CPU (prevents VRAM contention with TTS)")

    try:
        device = "cpu"
        compute_type = "int8"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logging.info(f"✅ Whisper {model_size} loaded on CPU")
        return model, "cpu"
    except Exception as e:
        logging.error(f"Failed to load Whisper model on CPU: {e}")
        raise


def _initialize_worker():
    """Load Whisper model once per worker process."""
    global _model, _device
    if _model is not None:
        return

    try:
        _model, _device = _load_whisper_model()
        logging.info(f"Worker {os.getpid()} loaded ASR model on {_device}")
    except Exception as e:
        error_msg = f"Worker {os.getpid()} failed to load ASR model: {e}"
        logging.error(error_msg)
        # Re-raise to signal failure
        raise


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def _validate_chunk(task_data: dict) -> dict:
    """Validate a single chunk (runs in worker with persistent model)."""
    global _model

    if _model is None:
        _initialize_worker()

    chunk_id = task_data['chunk_id']
    wav_path = Path(task_data['wav_path'])
    expected_text = task_data['expected_text']
    threshold = task_data['threshold']

    # Wait for file (handle async save)
    max_wait = 30
    start = time.time()
    while not wav_path.exists() and (time.time() - start) < max_wait:
        time.sleep(0.1)

    if not wav_path.exists():
        return {
            'chunk_id': chunk_id,
            'passed': False,
            'score': 0.0,
            'error': f'File not found: {wav_path}',
            'asr_text': ''
        }

    try:
        # Transcribe with persistent model
        segments, _ = _model.transcribe(
            str(wav_path),
            language="en",  # Force English - audio is always English TTS
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )
        asr_text = " ".join([seg.text for seg in segments]).strip()

        # Compute similarity
        expected_norm = _normalize_text(expected_text)
        asr_norm = _normalize_text(asr_text)
        score = SequenceMatcher(None, expected_norm, asr_norm).ratio()

        return {
            'chunk_id': chunk_id,
            'passed': score >= threshold,
            'score': score,
            'asr_text': asr_text,
            'expected_text': expected_text,
            'error': None
        }
    except Exception as e:
        return {
            'chunk_id': chunk_id,
            'passed': False,
            'score': 0.0,
            'error': str(e),
            'asr_text': ''
        }


def run_daemon(queue_dir: Path, results_dir: Path, num_workers: int = 4):
    """Main daemon loop: watch queue, process tasks, write results."""
    # Setup logging to file (since stdout/stderr might be redirected to DEVNULL)
    log_file = queue_dir.parent / "asr_daemon.log"
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=str(log_file),
            filemode='w',
            force=True  # Override any existing config
        )
    except Exception as e:
        # Fallback to basic config if file logging fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            force=True
        )

    logging.info(f"🚀 ASR daemon starting (PID: {os.getpid()})")
    logging.info(f"🖥️  Queue dir: {queue_dir}")
    logging.info(f"📁 Results dir: {results_dir}")
    logging.info(f"👷 Workers: {num_workers}")
    logging.info(f"📝 Log file: {log_file}")

    # Ensure parent directories exist before creating subdirectories
    try:
        queue_dir.mkdir(exist_ok=True, parents=True)
        results_dir.mkdir(exist_ok=True, parents=True)
        logging.info(f"✅ Created directories")
    except Exception as e:
        error_msg = f"❌ Failed to create directories: {e}"
        logging.error(error_msg)
        raise

    # Write PID file BEFORE creating executor (which blocks on worker initialization)
    pid_file = queue_dir.parent / "asr_daemon.pid"
    try:
        pid_file.write_text(f"{os.getpid()}\n{time.time()}")
        logging.info(f"📄 PID file written: {pid_file}")
    except Exception as e:
        logging.error(f"❌ Failed to write PID file: {e}")
        raise

    logging.info(f"🚀 Creating ProcessPoolExecutor with {num_workers} workers...")
    try:
        executor = ProcessPoolExecutor(max_workers=num_workers, initializer=_initialize_worker)
        logging.info(f"✅ ProcessPoolExecutor initialized with {num_workers} workers")
    except Exception as e:
        logging.error(f"❌ Failed to initialize ProcessPoolExecutor: {e}")
        raise
    pending_futures = {}  # future -> chunk_id

    try:
        while True:
            # Update heartbeat
            pid_file.write_text(f"{os.getpid()}\n{time.time()}")

            # Submit new tasks
            for task_file in sorted(queue_dir.glob("*.task.json")):
                try:
                    task_data = json.loads(task_file.read_text())
                    chunk_id = task_data['chunk_id']

                    # Submit to worker pool
                    future = executor.submit(_validate_chunk, task_data)
                    pending_futures[future] = chunk_id

                    # Remove task file
                    task_file.unlink()
                    logging.info(f"Queued {chunk_id}")
                except Exception as e:
                    logging.error(f"Failed to process task {task_file}: {e}")

            # Collect completed results
            done_futures = [f for f in pending_futures if f.done()]
            for future in done_futures:
                chunk_id = pending_futures.pop(future)
                try:
                    result = future.result()
                    result_file = results_dir / f"{chunk_id}.result.json"
                    result_file.write_text(json.dumps(result, indent=2))
                    logging.info(f"Completed {chunk_id} (score: {result['score']:.3f})")
                except Exception as e:
                    logging.error(f"Error processing {chunk_id}: {e}")

            # Check for shutdown signal
            shutdown_file = queue_dir.parent / "asr_daemon.shutdown"
            if shutdown_file.exists():
                logging.info("Shutdown signal received")
                shutdown_file.unlink()
                break

            time.sleep(0.1)  # Polling interval

    finally:
        executor.shutdown(wait=True)
        pid_file.unlink(missing_ok=True)
        logging.info("ASR daemon stopped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run_daemon(args.queue_dir, args.results_dir, args.workers)
