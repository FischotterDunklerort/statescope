"""Optional lossless representation of already-issued evidence, independent of auditing."""
from array import array
from base64 import b64encode, b64decode
import sys

FORMAT = "statescope.member-dictionary.v1"
MARK = "$statescope.indices"

def pack(record):
    ids, index = [], {}
    def visit(value, key=None):
        if isinstance(value, dict):
            if MARK in value:
                raise ValueError("reserved dictionary marker in input")
            return {k: visit(v, k) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            if value and key and (key in ("source_ids", "upper_bound") or "member" in key) and all(type(s) is str for s in value):
                refs = array("I")
                if refs.itemsize != 4:
                    raise ValueError("unsupported integer width")
                for s in value:
                    if s not in index:
                        index[s] = len(ids)
                        ids.append(s)
                    refs.append(index[s])
                if sys.byteorder != "little":
                    refs.byteswap()
                return {MARK: b64encode(refs.tobytes()).decode("ascii")}
            return [visit(v) for v in value]
        return value
    tree = visit(record)
    return {"storage_format": FORMAT, "identities": ids, "record": tree}

def unpack(container):
    if "storage_format" not in container:
        return container
    if set(container) != {"storage_format", "identities", "record"} or container["storage_format"] != FORMAT:
        raise ValueError("invalid storage envelope")
    ids = container["identities"]
    if not isinstance(ids,list) or any(type(s) is not str for s in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid member dictionary")
    seen = set()
    def visit(value, key=None):
        if isinstance(value, dict):
            if MARK in value:
                if set(value) != {MARK} or not key or not (key in ("upper_bound","source_ids") or "member" in key):
                    raise ValueError("invalid member reference field")
                refs = array("I")
                refs.frombytes(b64decode(value[MARK],validate=True))
                if refs.itemsize != 4 or not refs:
                    raise ValueError("invalid member reference width or length")
                if sys.byteorder != "little":
                    refs.byteswap()
                if max(refs) >= len(ids):
                    raise ValueError("member reference out of range")
                seen.update(refs)
                return [ids[i] for i in refs]
            return {k:visit(v,k) for k,v in value.items()}
        if isinstance(value,list):
            return [visit(v) for v in value]
        return value
    record=visit(container["record"])
    if len(seen)!=len(ids):
        raise ValueError("unreferenced member identities")
    return record
