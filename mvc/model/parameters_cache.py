from threading import Lock
from typing import Dict


class ParametersCache:
    def __init__(self):
        self._cache = {}
        self._lock = Lock()
    
    def get(self, userId: str):
        with self._lock:
            return self._cache.get(userId)
    
    def set(self, userId: str, parameters: Dict):
        with self._lock:
            self._cache[userId] = parameters
    
    def remove(self, userId: str):
        with self._lock:
            if userId in self._cache:
                self._cache.pop(userId, None)
                
    def clearCache(self):
        with self._lock:
            self._cache = {}
            
            
parameters_cache = ParametersCache()