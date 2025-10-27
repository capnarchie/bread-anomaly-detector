"""
Relay Controller Module
Handles relay activation with threading and queue management
"""
import queue
import threading
import time


class RelayController:
    """Manages relay activation with thread-safe queueing"""
    
    def __init__(self):
        self.relay_queue = queue.Queue()
        self.relay_active = False
        self.relay_lock = threading.Lock()
        self.activated_bread_ids = set()
        
        # Start worker thread
        self.worker_thread = threading.Thread(target=self._worker_thread, daemon=True)
        self.worker_thread.start()
    
    def _worker_thread(self):
        """Single worker thread that processes relay activation requests"""
        while True:
            try:
                # Wait for a relay activation request (blocking)
                relay_number, duration = self.relay_queue.get(timeout=1)
                
                with self.relay_lock:
                    if self.relay_active:
                        self.relay_queue.task_done()
                        continue  # Skip if relay is already active
                    self.relay_active = True
                
                try:
                    time.sleep(0.1)
                    print(f'Activating relay {relay_number} for {duration}s')
                    # Uncomment when relay module is available:
                    # from relay import on_relay, off_relay
                    # on_relay(relay_number)
                    time.sleep(duration)
                    # off_relay(relay_number)
                    print(f'Relay {relay_number} deactivated')
                finally:
                    with self.relay_lock:
                        self.relay_active = False
                    self.relay_queue.task_done()
                    
            except queue.Empty:
                continue  # No requests, continue waiting
            except Exception as e:
                print(f"Error in relay worker thread: {e}")
                continue
    
    def activate(self, relay_number=1, duration=0.1, bread_id=None):
        """
        Queue a relay activation request (non-blocking)
        
        Args:
            relay_number: The relay to activate
            duration: How long to keep the relay on (in seconds)
            bread_id: The ID of the bread triggering the relay (optional, for tracking)
        
        Returns:
            bool: True if request was queued, False if skipped
        """
        # Check if this bread ID has already activated the relay
        if bread_id is not None:
            if bread_id in self.activated_bread_ids:
                return False  # Skip if this bread ID has already activated the relay
            self.activated_bread_ids.add(bread_id)  # Mark this bread ID as activated
        
        with self.relay_lock:
            if self.relay_active:
                return False  # Skip if relay is already active
        
        # Add request to queue (non-blocking)
        self.relay_queue.put((relay_number, duration))
        return True
    
    def reset_bread_id(self, bread_id):
        """Remove bread ID from activated set (for testing/reset purposes)"""
        self.activated_bread_ids.discard(bread_id)
    
    def clear_activated_ids(self):
        """Clear all activated bread IDs (for testing/reset purposes)"""
        self.activated_bread_ids.clear()
