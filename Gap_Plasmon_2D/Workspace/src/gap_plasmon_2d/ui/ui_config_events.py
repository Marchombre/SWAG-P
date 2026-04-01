# ui_config_events.py
import threading
import weakref


class _WeakCallback:
    """
    Encapsule un callback de manière faible pour éviter les fuites mémoire.
    """

    def __init__(self, callback):
        try:
            self._ref = weakref.WeakMethod(callback)
        except TypeError:
            self._ref = weakref.ref(callback)

    def get(self):
        return self._ref()

    def is_alive(self):
        return self.get() is not None


class _EventBus:
    """
    Petit bus d'événements générique, léger et thread-safe.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers = []

    def subscribe(self, callback):
        wrapped = _WeakCallback(callback)
        with self._lock:
            self._cleanup_locked()
            self._subscribers.append(wrapped)

        def unsubscribe():
            self.unsubscribe(callback)

        return unsubscribe

    def unsubscribe(self, callback):
        with self._lock:
            alive = []
            for sub in self._subscribers:
                cb = sub.get()
                if cb is None:
                    continue
                if cb is callback:
                    continue
                alive.append(sub)
            self._subscribers = alive

    def notify(self):
        callbacks = []
        with self._lock:
            self._cleanup_locked()
            for sub in self._subscribers:
                cb = sub.get()
                if cb is not None:
                    callbacks.append(cb)

        for cb in callbacks:
            try:
                cb()
            except Exception as e:
                print(f"[ui_config_events] erreur callback : {e}")

    def _cleanup_locked(self):
        self._subscribers = [sub for sub in self._subscribers if sub.is_alive()]


_GEOM_MAT_CONFIG_EVENT_BUS = _EventBus()


def subscribe_geom_mat_configs_changed(callback):
    """
    Abonne un callback à l'événement 'les configs geom+mat ont changé'.
    """
    return _GEOM_MAT_CONFIG_EVENT_BUS.subscribe(callback)


def unsubscribe_geom_mat_configs_changed(callback):
    """
    Désabonne explicitement un callback.
    """
    _GEOM_MAT_CONFIG_EVENT_BUS.unsubscribe(callback)


def notify_geom_mat_configs_changed():
    """
    Notifie tous les abonnés qu'une config geom+mat a changé.
    """
    _GEOM_MAT_CONFIG_EVENT_BUS.notify()