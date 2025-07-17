# file_watchers.py
import threading
import re
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from IPython import get_ipython

class DebouncedEventHandler(FileSystemEventHandler):
    """
    FileSystemEventHandler qui regroupe les événements par lot
    et n'appelle la callback qu'une seule fois après `debounce_interval`.
    Peut filtrer par extensions ou par patterns regex.
    """
    def __init__(self, callback, *, debounce_interval=0.1, extensions=None, patterns=None):
        """
        callback          : fonction callable sans arguments.
        debounce_interval : délai (en s) après le dernier événement avant d'appeler callback.
        extensions        : liste de suffixes ('.json', '.h5', ...) à filtrer.
        patterns          : liste de regex (string ou compiled) pour matcher le path.
                            Si spécifié, c'est OR sur toutes les regex.
        """
        super().__init__()
        self.callback = callback
        self.debounce_interval = debounce_interval
        self.extensions = extensions
        self.patterns = [re.compile(p) for p in patterns] if patterns else None
        self._lock = threading.Lock()
        self._timer = None

    def _match(self, path: str) -> bool:
        if self.extensions and not any(path.endswith(ext) for ext in self.extensions):
            return False
        if self.patterns and not any(p.search(path) for p in self.patterns):
            return False
        return True

    def _schedule(self):
        with self._lock:
            # annule l'ancien timer s'il existe
            if self._timer is not None:
                self._timer.cancel()
            # crée un nouveau timer
            self._timer = threading.Timer(self.debounce_interval, self._run_callback)
            self._timer.daemon = True
            self._timer.start()

    def _run_callback(self):
        with self._lock:
            self._timer = None
        def _cb():
            try:
                self.callback()
            except Exception as e:
                print(f"[watcher] erreur dans callback : {e}")
        get_ipython().kernel.io_loop.add_callback(_cb)

    # On traite tous les événements de fichiers
    def on_created(self, event): self._handle(event)
    def on_deleted(self, event): self._handle(event)
    def on_modified(self, event): self._handle(event)
    def on_moved(self, event):   self._handle(event, dest=True)

    def _handle(self, event, dest=False):
        path = event.dest_path if dest else event.src_path
        if event.is_directory:
            return
        if self._match(path):
            self._schedule()

    def stop(self):
        """Annule le timer si nécessaire."""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None

def start_watcher(path, callback, *,
                  debounce_interval=0.1,
                  extensions=None,
                  patterns=None,
                  recursive=False):
    """
    Démarre un Observer sur `path` avec un DebouncedEventHandler.
    Retourne (observer, handler) pour permettre un arrêt propre.
    """
    handler = DebouncedEventHandler(
        callback,
        debounce_interval=debounce_interval,
        extensions=extensions,
        patterns=patterns
    )
    obs = Observer()
    obs.schedule(handler, path=path, recursive=recursive)
    obs.daemon = True
    obs.start()
    return obs, handler
