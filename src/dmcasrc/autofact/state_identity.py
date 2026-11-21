"""Live object generations; content equality never establishes origin."""
import secrets
from .identity_evidence import IdentityEvidenceMap


class IdentityRegistry:
    def __init__(self): self._objects = IdentityEvidenceMap()

    def state(self, obj):
        item = self._objects.get(id(obj))
        if item is None or item[0] is not obj:
            item = (obj, secrets.token_hex(16), 0)
            self._objects[id(obj)] = item
        return item[1], item[2]

    def uid(self, obj): return self.state(obj)[0]

    def advance(self, obj):
        uid, generation = self.state(obj)
        self._objects[id(obj)] = (obj, uid, generation + 1)


LIVE_STATES = IdentityRegistry()
