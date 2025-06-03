from threading import Lock
from extensions import socketio


class SocketCache:
    def __init__(self):
        self._cache: dict = {}
        self._lock = Lock()

    def get(self, userId: str):
        with self._lock:
            return self._cache.get(userId)

    def set(self, userId: str, sid: dict):
        with self._lock:
            self._cache[userId] = sid

    def remove(self, userId: str):
        with self._lock:
            if userId in self._cache:
                self._cache.pop(userId, None)

    def emit(self, userId: str, event: str, data: dict):
        with self._lock:
            sid = self._cache[userId]["sid"] if userId in self._cache else None,
            if sid:
                socketio.emit(
                    event,
                    data,
                    to=sid,
                    namespace="/train/train",
                )

    def clearCache(self):
        with self._lock:
            self._cache = {}


socket_cache = SocketCache()
