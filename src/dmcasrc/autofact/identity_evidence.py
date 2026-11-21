"""Live-object identity evidence without keeping disposable data alive.

Historical source facts live in runtime events. This map holds only the live
lookup needed to relate a still-existing data object to those facts. Objects
without weak-reference support keep the previous strong-reference behavior.
"""
from collections.abc import MutableMapping
import weakref

class IdentityEvidenceMap(MutableMapping):
    def __init__(self):
        self._entries = {}

    def __setitem__(self, key, value):
        obj, *tail = value
        owner = weakref.ref(self)
        def expired(ref):
            mapping = owner()
            if mapping is not None:
                current = mapping._entries.get(key)
                if current is not None and current[0] is ref:
                    mapping._entries.pop(key, None)
        try:
            ref = weakref.ref(obj, expired)
        except TypeError:
            self._entries[key] = (None, tuple(value))
        else:
            self._entries[key] = (ref, tuple(tail))

    def __getitem__(self, key):
        ref, payload = self._entries[key]
        if ref is None:
            return payload
        obj = ref()
        if obj is None:
            self._entries.pop(key, None)
            raise KeyError(key)
        return (obj, *payload)

    def __delitem__(self, key):
        del self._entries[key]

    def __iter__(self):
        return iter(tuple(self._entries))

    def __len__(self):
        return len(self._entries)
