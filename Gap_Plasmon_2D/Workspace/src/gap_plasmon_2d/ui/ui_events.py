# ui_events.py
import threading
import weakref


class _WeakCallback:
    """
    Encapsule un callback de manière faible pour éviter les fuites mémoire.
    - fonction libre  -> weakref.ref
    - méthode liée    -> weakref.WeakMethod
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


class GeometryEventBus:
    """
    Bus d’événements léger pour notifier les changements de géométrie.
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
            new_subscribers = []
            for sub in self._subscribers:
                cb = sub.get()
                if cb is None:
                    continue
                if cb is callback:
                    continue
                new_subscribers.append(sub)
            self._subscribers = new_subscribers

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
                print(f"[ui_events] erreur dans callback geometry_changed : {e}")

    def _cleanup_locked(self):
        self._subscribers = [sub for sub in self._subscribers if sub.is_alive()]


_GEOMETRY_EVENT_BUS = GeometryEventBus()


def subscribe_geometry_changed(callback):
    """
    Abonne un callback callable sans argument à l’événement
    'une géométrie a changé'.
    Retourne une fonction unsubscribe().
    """
    return _GEOMETRY_EVENT_BUS.subscribe(callback)


def unsubscribe_geometry_changed(callback):
    """
    Désabonne explicitement un callback.
    """
    _GEOMETRY_EVENT_BUS.unsubscribe(callback)


def notify_geometry_changed():
    """
    Notifie que les géométries sauvegardées ont changé.
    """
    _GEOMETRY_EVENT_BUS.notify()