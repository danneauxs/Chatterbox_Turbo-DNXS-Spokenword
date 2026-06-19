"""
Real-Time Status Manager for TTS Processing
============================================

Provides simple, reliable real-time status updates during TTS processing.
Thread-safe GUI updates using Qt signals.

Author: OpenCode
Date: 2025-12-29
"""

import time
from datetime import timedelta

from PyQt5.QtCore import QObject, pyqtSignal


class RealTimeStatusManager(QObject):
    """
    Manages real-time status tracking and GUI updates for TTS processing.
    
    Tracks timing, calculates metrics, and updates GUI using Qt signals for thread safety.
    """
    
    # Signal to update GUI from any thread
    update_signal = pyqtSignal(float, float, float, int, int, float)
    
    def __init__(self, status_panel=None):
        """
        Initialize status manager.
        
        Args:
            status_panel: GUI status panel widget to update (optional)
        """
        super().__init__()
        self.status_panel = status_panel
        self.start_time = None
        self.chunk_times = []
        self.total_audio_duration = 0.0
        self.total_chunks = 0
        self.vram_usage = 0.0
        
        # Connect signal to update method if panel exists
        if self.status_panel:
            self.update_signal.connect(self._update_panel_safe)
        
    def on_conversion_start(self, total_chunks):
        """
        Called when conversion starts.
        
        Args:
            total_chunks (int): Total number of chunks to process
        """
        self.start_time = time.time()
        self.chunk_times = []
        self.total_audio_duration = 0.0
        self.total_chunks = total_chunks
        
        if self.status_panel:
            self.status_panel.reset()
            
    def on_chunk_complete(self, chunk_idx, total_chunks, chunk_audio_duration=0.0, vram_usage=None):
        """
        Called when a chunk completes processing.
        
        Args:
            chunk_idx (int): Index of completed chunk (0-based)
            total_chunks (int): Total number of chunks
            chunk_audio_duration (float): Duration of generated audio in seconds
            vram_usage (str, optional): VRAM usage string (e.g., "3.2 GB")
        """
        if not self.start_time:
            self.on_conversion_start(total_chunks)
            
        # Record completion time
        self.chunk_times.append(time.time())
        self.total_audio_duration += chunk_audio_duration
        
        if vram_usage:
            self.vram_usage = vram_usage
        
        # Calculate metrics
        elapsed = time.time() - self.start_time
        completed_count = len(self.chunk_times)
        avg_chunk_time = elapsed / completed_count if completed_count > 0 else 0
        remaining_chunks = total_chunks - completed_count
        estimated_remaining = avg_chunk_time * remaining_chunks
        
        # Calculate realtime factor
        if self.total_audio_duration > 0 and elapsed > 0:
            realtime_factor = self.total_audio_duration / elapsed
        else:
            realtime_factor = 0.0
            
        # Update GUI using signal (thread-safe)
        if self.status_panel:
            self.update_signal.emit(
                elapsed,
                estimated_remaining,
                realtime_factor,
                completed_count,
                total_chunks,
                self.vram_usage
            )
    
    def _update_panel_safe(self, elapsed_seconds, remaining_seconds, realtime_factor, completed_chunks, total_chunks, vram_usage):
        """Thread-safe GUI update method called via signal"""
        if self.status_panel:
            self.status_panel.update_realtime(
                elapsed_seconds=elapsed_seconds,
                remaining_seconds=remaining_seconds,
                realtime_factor=realtime_factor,
                completed_chunks=completed_chunks,
                total_chunks=total_chunks,
                vram_usage=vram_usage
            )
            
    def on_conversion_complete(self):
        """Called when conversion completes."""
        if self.status_panel and self.start_time:
            elapsed = time.time() - self.start_time
            
            # Final update with exact values
            if self.total_audio_duration > 0:
                realtime_factor = self.total_audio_duration / elapsed
            else:
                realtime_factor = 0.0
                
            # Use signal for thread-safe update
            self.update_signal.emit(
                elapsed,
                0.0,
                realtime_factor,
                self.total_chunks,
                self.total_chunks,
                self.vram_usage
            )
            
    def get_status_summary(self):
        """
        Get current status as a dictionary.
        
        Returns:
            dict: Status metrics (elapsed, remaining, realtime, etc.)
        """
        if not self.start_time:
            return {}
            
        elapsed = time.time() - self.start_time
        completed_count = len(self.chunk_times)
        avg_chunk_time = elapsed / completed_count if completed_count > 0 else 0
        remaining_chunks = self.total_chunks - completed_count
        estimated_remaining = avg_chunk_time * remaining_chunks
        realtime_factor = self.total_audio_duration / elapsed if elapsed > 0 and self.total_audio_duration > 0 else 0.0
        
        return {
            'elapsed': elapsed,
            'remaining': estimated_remaining,
            'realtime': realtime_factor,
            'completed': completed_count,
            'total': self.total_chunks,
            'audio_duration': self.total_audio_duration,
            'vram': self.vram_usage
        }


# Global status manager instance for access from processing functions
_global_status_manager = None


def get_status_manager():
    """Get the global status manager instance."""
    global _global_status_manager
    return _global_status_manager


def set_status_manager(manager):
    """Set the global status manager instance."""
    global _global_status_manager
    _global_status_manager = manager
