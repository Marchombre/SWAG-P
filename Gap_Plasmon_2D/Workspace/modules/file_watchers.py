# file_watchers.py
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from IPython import get_ipython

class RefreshEventHandler(FileSystemEventHandler):
    """
    Handler générique : appelle self.callback() dès qu’un fichier .h5
    est créé, modifié, supprimé ou déplacé dans le dossier observé.
    """
    def __init__(self, callback, extensions=None):
        """
        callback   : fonction sans argument à appeler lors d'un event
        extensions : liste d’extensions à filtrer (e.g. ['.h5']), ou None pour tout
        """
        super().__init__()
        self.callback = callback
        self.extensions = extensions

    def _should_handle(self, path):
        if not self.extensions:
            return True
        return any(path.endswith(ext) for ext in self.extensions)

    # On gère tous les types d’événements
    def on_created(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            self._schedule_update()

    def on_deleted(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            self._schedule_update()

    def on_modified(self, event):
        if not event.is_directory and self._should_handle(event.src_path):
            self._schedule_update()

    def on_moved(self, event):
        # moved: on regarde le path de destination
        if not event.is_directory and self._should_handle(event.dest_path):
            self._schedule_update()

    def _schedule_update(self):
        # Pour exécuter la callback dans le thread principal IPython
        def _cb():
            try:
                self.callback()
            except Exception as e:
                print(f"[watcher] erreur dans callback : {e}")
        get_ipython().kernel.io_loop.add_callback(_cb)

def start_watcher(path, callback, extensions=None, recursive=False):
    """
    Démarre un Observer sur `path`, avec un RefreshEventHandler.
    Retourne l’Observer, pour pouvoir l’arrêter plus tard (obs.stop()).
    """
    handler = RefreshEventHandler(callback, extensions=extensions)
    obs = Observer()
    obs.schedule(handler, path=path, recursive=recursive)
    obs.daemon = True
    obs.start()
    return obs