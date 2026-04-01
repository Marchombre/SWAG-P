import threading
import weakref


class _WeakCallback:
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
                print(f"[ui_material_events] erreur callback : {e}")

    def _cleanup_locked(self):
        self._subscribers = [sub for sub in self._subscribers if sub.is_alive()]


_MATERIAL_CONFIG_EVENT_BUS = _EventBus()


def subscribe_material_config_changed(callback):
    return _MATERIAL_CONFIG_EVENT_BUS.subscribe(callback)


def unsubscribe_material_config_changed(callback):
    _MATERIAL_CONFIG_EVENT_BUS.unsubscribe(callback)


def notify_material_config_changed():
    _MATERIAL_CONFIG_EVENT_BUS.notify()