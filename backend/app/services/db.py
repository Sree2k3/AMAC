import json, pathlib

DB_PATH = pathlib.Path(__file__).parent / "cache.json"

def _load() -> dict:
    if not DB_PATH.exists():
        return {}
    return json.loads(DB_PATH.read_text(encoding="utf-8") or "{}")

def _save(data: dict) -> None:
    DB_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

def get(key: str):
    """Return the value stored under *key* (or `None`)."""
    return _load().get(key, None)

def set(key, value):
    data = _load()
    data[key] = value
    _save(data)

def delete(key):
    data = _load()
    data.pop(key, None)
    _save(data)

