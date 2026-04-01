# file_watchers.py
import os
import errno
import re
import threading

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from IPython import get_ipython


# =====================================================================
# Handler debounced avec filtrage par extensions / regex / fichiers exacts
# =====================================================================

class DebouncedEventHandler(FileSystemEventHandler):
    """
    FileSystemEventHandler qui regroupe les événements par lot
    et n'appelle la callback qu'une seule fois après `debounce_interval`.

    Peut filtrer :
      - par extensions
      - par patterns regex
      - par chemins de fichiers exacts
    """

    def __init__(
        self,
        callback,
        *,
        debounce_interval=0.1,
        extensions=None,
        patterns=None,
        exact_paths=None,
    ):
        super().__init__()
        self.callback = callback
        self.debounce_interval = debounce_interval
        self.extensions = tuple(extensions or [])
        self.patterns = [re.compile(p) for p in patterns] if patterns else None
        self.exact_paths = {
            os.path.abspath(p) for p in (exact_paths or [])
        } if exact_paths else None

        self._lock = threading.Lock()
        self._timer = None

    def _normalize(self, path: str) -> str:
        return os.path.abspath(path)

    def _match(self, path: str) -> bool:
        normalized = self._normalize(path)

        if self.exact_paths is not None and normalized not in self.exact_paths:
            return False

        if self.extensions and not normalized.endswith(self.extensions):
            return False

        if self.patterns and not any(p.search(normalized) for p in self.patterns):
            return False

        return True

    def _schedule(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()

            self._timer = threading.Timer(
                self.debounce_interval,
                self._run_callback
            )
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

        ip = get_ipython()
        if ip is not None and getattr(ip, "kernel", None) is not None:
            ip.kernel.io_loop.add_callback(_cb)
        else:
            # Hors Jupyter, fallback direct
            _cb()

    def on_created(self, event):
        self._handle(event, dest=False)

    def on_deleted(self, event):
        self._handle(event, dest=False)

    def on_modified(self, event):
        self._handle(event, dest=False)

    def on_moved(self, event):
        self._handle(event, dest=True)

    def _handle(self, event, dest=False):
        if event.is_directory:
            return

        path = event.dest_path if dest else event.src_path
        if self._match(path):
            self._schedule()

    def stop(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


# =====================================================================
# Observer global + fallback polling
# =====================================================================

_GLOBAL_OBSERVER = None
_GLOBAL_OBSERVER_MODE = None   # "inotify" ou "polling"
_GLOBAL_OBSERVER_LOCK = threading.Lock()


def _create_inotify_observer():
    obs = Observer()
    obs.daemon = True
    obs.start()
    return obs


def _create_polling_observer():
    obs = PollingObserver()
    obs.daemon = True
    obs.start()
    return obs


def get_global_observer():
    global _GLOBAL_OBSERVER, _GLOBAL_OBSERVER_MODE

    with _GLOBAL_OBSERVER_LOCK:
        if _GLOBAL_OBSERVER is not None:
            return _GLOBAL_OBSERVER

        try:
            obs = _create_inotify_observer()
            _GLOBAL_OBSERVER = obs
            _GLOBAL_OBSERVER_MODE = "inotify"
            print("[watcher] ✅ Observer global créé en mode inotify.")
            return obs
        except OSError as e:
            if e.errno in (errno.EMFILE, errno.ENOSPC, 24):
                print(
                    "[watcher] ⚠️ Limite inotify atteinte. "
                    "Bascule automatique en mode polling."
                )
            else:
                raise

        try:
            obs = _create_polling_observer()
            _GLOBAL_OBSERVER = obs
            _GLOBAL_OBSERVER_MODE = "polling"
            print("[watcher] ✅ Observer global créé en mode polling.")
            return obs
        except Exception as e:
            print(
                "[watcher] ❌ Impossible de créer un observer, même en polling.\n"
                f"         Détail : {e}"
            )
            _GLOBAL_OBSERVER = None
            _GLOBAL_OBSERVER_MODE = None
            return None


def stop_global_observer():
    global _GLOBAL_OBSERVER, _GLOBAL_OBSERVER_MODE

    with _GLOBAL_OBSERVER_LOCK:
        if _GLOBAL_OBSERVER is not None:
            try:
                _GLOBAL_OBSERVER.stop()
                _GLOBAL_OBSERVER.join(timeout=1.0)
            except Exception as e:
                print(f"[watcher] erreur à l'arrêt de l'observer global : {e}")

        _GLOBAL_OBSERVER = None
        _GLOBAL_OBSERVER_MODE = None


# =====================================================================
# API publique
# =====================================================================

def start_watcher(
    path,
    callback,
    *,
    debounce_interval=0.1,
    extensions=None,
    patterns=None,
    recursive=False,
):
    """
    Démarre un watcher partagé.

    Si `path` est un fichier :
      - on surveille son dossier parent
      - on filtre exactement ce fichier

    Si `path` est un dossier :
      - on surveille ce dossier directement
      - on applique extensions/patterns normalement
    """

    abs_path = os.path.abspath(path)

    if os.path.isfile(abs_path):
        watch_dir = os.path.dirname(abs_path)
        exact_paths = [abs_path]
    else:
        watch_dir = abs_path
        exact_paths = None

    handler = DebouncedEventHandler(
        callback,
        debounce_interval=debounce_interval,
        extensions=extensions,
        patterns=patterns,
        exact_paths=exact_paths,
    )

    obs = get_global_observer()
    if obs is None:
        print(
            "[watcher] ⚠️ Aucun observer disponible. "
            "Les mises à jour automatiques sont désactivées pour cette session."
        )
        return None, None

    obs.schedule(handler, path=watch_dir, recursive=recursive)

    mode = _GLOBAL_OBSERVER_MODE or "unknown"
    print(
        f"[watcher] ✅ Watcher démarré sur « {watch_dir} » "
        f"(cible initiale : {abs_path}, mode global : {mode}, recursive={recursive})."
    )

    return obs, handler