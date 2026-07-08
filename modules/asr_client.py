#!/usr/bin/env python3
"""
ASR Client Module - Communicates with ASR daemon for concurrent validation.

This module provides non-blocking ASR submission and result collection,
enabling the 5-phase concurrent ASR validation pipeline:
1. Generate chunks + concurrent ASR submission
2. Collect ASR results after generation
3. Regenerate failed chunks with multi-attempt best selection
4. Final validation
5. Concatenation + M4B creation
"""

import os
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class ASRClient:
    """ASR client for concurrent chunk validation via ASR daemon."""
    
    def __init__(self, tts_dir: Path, threshold: float):
        """
        Initialize ASR client.

        Args:
            tts_dir: TTS output directory
            threshold: ASR similarity threshold for validation
        """
        self.tts_dir = Path(tts_dir).resolve()  # Ensure absolute path
        self.threshold = threshold
        self.queue_dir = self.tts_dir / "asr_queue"
        self.results_dir = self.tts_dir / "asr_results"
        self.pending_chunks: Set[str] = set()
        self.daemon_process = None
        self.daemon_pid = None
        
        # Ensure directories exist
        self.queue_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def start_daemon(self) -> bool:
        """
        Start ASR daemon if not already running.

        Returns:
            bool: True if daemon started successfully or already running
        """
        if self._is_daemon_running():
            print(f"🔍 ASR daemon already running (PID: {self.daemon_pid})")
            return True

        try:
            # Locate ASR daemon script relative to this module
            project_root = Path(__file__).resolve().parent.parent
            asr_daemon = project_root / 'ASR' / 'asr_daemon.py'

            if not asr_daemon.exists():
                print(f"❌ ASR daemon script not found at: {asr_daemon}")
                return False

            print("🚀 Starting ASR daemon for concurrent validation...")

            # Build daemon command using current Python interpreter
            # This ensures the daemon uses the same venv as the main program
            cmd = [
                sys.executable,
                str(asr_daemon),
                '--queue-dir', str(self.queue_dir.resolve()),      # Absolute path
                '--results-dir', str(self.results_dir.resolve()),  # Absolute path
                '--workers', '4'
            ]
            
            # Launch daemon subprocess
            # Note: Use DEVNULL to avoid pipe buffering issues that could cause hangs
            # The daemon logs to its own stdout/stderr, which isn't critical for functionality
            popen_kwargs = {
                'cwd': str(self.tts_dir),
                'stdout': subprocess.DEVNULL,
                'stderr': subprocess.DEVNULL,
                'text': True,
            }

            # Cross-platform detached process
            if sys.platform == 'win32':
                popen_kwargs['creationflags'] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs['start_new_session'] = True

            self.daemon_process = subprocess.Popen(cmd, **popen_kwargs)

            # Wait for daemon to initialize and write PID file
            # Increased from 2→5 to account for:
            # - Python venv interpreter startup
            # - Module imports (faster-whisper, torch, etc.)
            # - ProcessPoolExecutor worker initialization
            # - CPU resource contention with TTS models
            time.sleep(10)
            
            # Check if daemon started successfully
            if self._is_daemon_running():
                print(f"✅ ASR daemon started successfully (PID: {self.daemon_pid})")
                return True
            else:
                print("❌ ASR daemon failed to start")

                # Provide detailed diagnostics
                log_file = self.tts_dir / "asr_daemon.log"

                if self.daemon_process:
                    if self.daemon_process.poll() is not None:
                        # Process has exited
                        print(f"   Process exited with code: {self.daemon_process.returncode}")
                        if log_file.exists():
                            print(f"   📝 Check log file: {log_file}")
                    else:
                        # Process still running but no PID file - daemon is stuck
                        print("   ⚠️ Daemon process is stuck (still running but no PID file)")
                        print("   Terminating unresponsive daemon...")

                        # Forcefully terminate the stuck process
                        try:
                            self.daemon_process.terminate()
                            # Give it 2 seconds to respond to SIGTERM
                            for i in range(20):  # 20 * 0.1 = 2 seconds
                                time.sleep(0.1)
                                if self.daemon_process.poll() is not None:
                                    print(f"   Daemon terminated (exit code: {self.daemon_process.returncode})")
                                    break
                            else:
                                # Still running after SIGTERM, force kill
                                self.daemon_process.kill()
                                time.sleep(0.5)
                                print("   Daemon forcefully killed")
                        except Exception as e:
                            print(f"   Could not terminate daemon: {e}")

                        # Show log file location and contents if available
                        log_file = self.tts_dir / "asr_daemon.log"
                        print(f"   📝 Check daemon log: {log_file}")
                        if log_file.exists():
                            try:
                                with open(log_file) as f:
                                    lines = f.readlines()
                                    if lines:
                                        print(f"   Last 5 log lines:")
                                        for line in lines[-5:]:
                                            print(f"      {line.rstrip()}")
                            except Exception as e:
                                print(f"   Could not read log file: {e}")

                        print("   ⚠️ Common issues:")
                        print("      • Insufficient RAM for Whisper model on CPU")
                        print("      • CPU resources exhausted (competing with TTS)")
                        print("      • faster-whisper package not installed or broken")
                        print("      • Directory permission issues")

                return False
                
        except Exception as e:
            print(f"❌ Failed to start ASR daemon: {e}")
            return False
    
    def submit(self, chunk_id: str, wav_path: Path, expected_text: str) -> None:
        """
        Submit a chunk for ASR validation (non-blocking).
        
        Args:
            chunk_id: Chunk identifier (e.g., 'chunk_00001')
            wav_path: Path to generated WAV file
            expected_text: Expected reference text
        """
        task_file = self.queue_dir / f"{chunk_id}.task.json"
        
        task_data = {
            'chunk_id': chunk_id,
            'wav_path': str(wav_path.resolve()),
            'expected_text': expected_text,
            'threshold': self.threshold,
            'timestamp': time.time()
        }
        
        try:
            task_file.write_text(json.dumps(task_data, indent=2))
            self.pending_chunks.add(chunk_id)
            logging.info(f"📤 Submitted {chunk_id} for ASR validation")
        except Exception as e:
            logging.error(f"Failed to submit {chunk_id} to ASR queue: {e}")
    
    def get_result(self, chunk_id: str, timeout: int = 120) -> Optional[Dict]:
        """
        Get ASR validation result for a chunk (blocking with timeout).
        
        Args:
            chunk_id: Chunk identifier
            timeout: Maximum seconds to wait for result
            
        Returns:
            dict: ASR result or None if timeout/error
        """
        result_file = self.results_dir / f"{chunk_id}.result.json"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if result_file.exists():
                try:
                    result = json.loads(result_file.read_text())
                    self.pending_chunks.discard(chunk_id)
                    logging.info(f"📥 Received result for {chunk_id} (score: {result.get('score', 0.0):.3f})")
                    return result
                except Exception as e:
                    logging.error(f"Failed to parse ASR result for {chunk_id}: {e}")
                    return None
            
            time.sleep(0.5)  # Poll every 500ms
        
        logging.warning(f"⏰ Timeout waiting for ASR result for {chunk_id}")
        return None
    
    def collect_all_results(self, expected_chunk_count: int = None) -> Tuple[List[Dict], List[str]]:
        """
        Collect results for all pending chunks.

        Args:
            expected_chunk_count: Total number of chunks that were supposed to be validated.
                                  Used to detect if ASR submission was silently skipped.

        Returns:
            tuple: (all_results, failed_chunks)
        """
        all_results = []
        failed_chunks = []

        print(f"⏳ Waiting for {len(self.pending_chunks)} ASR validation results...")

        # Guard: warn if no chunks were submitted but chunks were generated
        if len(self.pending_chunks) == 0 and expected_chunk_count and expected_chunk_count > 0:
            logging.warning(f"⚠️ ASR validation skipped: 0 of {expected_chunk_count} chunks were submitted — asr_client may not have been wired into the processing path")
            print(f"⚠️ ASR validation skipped: 0 of {expected_chunk_count} chunks were submitted — asr_client may not have been wired into the processing path")
            return all_results, failed_chunks

        for chunk_id in list(self.pending_chunks):
            result = self.get_result(chunk_id, timeout=60)
            if result:
                all_results.append(result)
                if not result.get('passed', False):
                    failed_chunks.append({
                        'chunk_id': chunk_id,
                        'score': result.get('score', 0.0),
                        'asr_text': result.get('asr_text', ''),
                        'expected_text': result.get('expected_text', ''),
                        'error': result.get('error', '')
                    })
                else:
                    logging.info(f"✅ {chunk_id} passed ASR validation (score: {result.get('score', 0.0):.3f})")
            else:
                logging.error(f"❌ No result for {chunk_id}, marking as failed")
                failed_chunks.append({
                    'chunk_id': chunk_id,
                    'score': 0.0,
                    'asr_text': '',
                    'expected_text': 'Unknown',
                    'error': 'Timeout or daemon error'
                })

        if failed_chunks:
            print(f"⚠️ {len(failed_chunks)} chunks failed initial ASR validation")
        else:
            print(f"✅ All {len(all_results)} chunks passed ASR validation")

        return all_results, failed_chunks
    
    def shutdown_daemon(self) -> bool:
        """
        Signal ASR daemon to shut down gracefully.
        
        Returns:
            bool: True if shutdown signal sent successfully
        """
        try:
            # Write shutdown signal file
            shutdown_file = self.queue_dir.parent / "asr_daemon.shutdown"
            shutdown_file.write_text("shutdown")
            print("📡 Sent shutdown signal to ASR daemon")
            
            # Wait for daemon to stop
            if self.daemon_process:
                try:
                    self.daemon_process.wait(timeout=10)
                    print("✅ ASR daemon stopped")
                except subprocess.TimeoutExpired:
                    print("⚠️ ASR daemon did not stop gracefully, forcing termination")
                    self.daemon_process.terminate()
                    time.sleep(2)
                    if self.daemon_process.poll() is None:
                        self.daemon_process.kill()
                        print("🔥 ASR daemon forced to stop")
            
            # Cleanup shutdown file
            if shutdown_file.exists():
                shutdown_file.unlink()
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to shutdown ASR daemon: {e}")
            return False
    
    def _is_daemon_running(self) -> bool:
        """Check if ASR daemon is running via PID file."""
        pid_file = self.queue_dir.parent / "asr_daemon.pid"
        
        if not pid_file.exists():
            self.daemon_pid = None
            return False
        
        try:
            pid_str = pid_file.read_text().strip()
            if '\n' in pid_str:
                # Format: "pid\nheartbeat"
                pid = int(pid_str.split('\n')[0])
            else:
                pid = int(pid_str)
            
            # Check if process exists
            try:
                os.kill(pid, 0)  # Signal 0 doesn't actually kill process
                self.daemon_pid = pid
                return True
            except OSError:
                # Process doesn't exist
                pid_file.unlink(missing_ok=True)
                self.daemon_pid = None
                return False
                
        except Exception as e:
            logging.warning(f"Error checking daemon status: {e}")
            return False
    
    def __enter__(self):
        """Context manager entry - start daemon."""
        if self.start_daemon():
            return self
        else:
            raise RuntimeError("Failed to start ASR daemon")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - shutdown daemon."""
        if self.pending_chunks:
            print(f"⚠️ {len(self.pending_chunks)} chunks still pending during shutdown")
        
        self.shutdown_daemon()
    
    def get_pending_count(self) -> int:
        """Get number of pending chunks."""
        return len(self.pending_chunks)