import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
HISTORY_LIMIT = 30


def _history_path() -> Path:
    override = os.environ.get("CRLSUITE_HISTORY_PATH")
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
    return base / "CRLSuite" / "config_history.json"


HISTORY_PATH = _history_path()


def save_configuration(path: Path, configuration: dict[str, Any]) -> None:
    document = {"schema_version": SCHEMA_VERSION, "configuration": configuration}
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def load_configuration(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read configuration file") from exc
    if not isinstance(document, dict):
        raise ValueError("Configuration file must contain a JSON object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported configuration schema version")
    configuration = document.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Configuration file does not contain a configuration")
    return configuration


def load_history(path: Path = HISTORY_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        return []
    entries = document.get("history", [])
    if not isinstance(entries, list):
        return []
    valid = []
    seen = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        timestamp = entry.get("timestamp")
        configuration = entry.get("configuration")
        if not isinstance(timestamp, str) or not isinstance(configuration, dict):
            continue
        if configuration in seen:
            continue
        seen.append(configuration)
        valid.append({"timestamp": timestamp, "configuration": configuration})
        if len(valid) == HISTORY_LIMIT:
            break
    return valid


def append_history(
    configuration: dict[str, Any], path: Path = HISTORY_PATH
) -> list[dict[str, Any]]:
    if not isinstance(configuration.get("stacks"), list) or not configuration["stacks"]:
        raise ValueError("History configuration requires at least one CRL stack")
    entries = load_history(path)
    entries = [
        entry for entry in entries if entry.get("configuration") != configuration
    ]
    entries.insert(
        0,
        {
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "configuration": configuration,
        },
    )
    entries = entries[:HISTORY_LIMIT]
    document = {"schema_version": SCHEMA_VERSION, "history": entries}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    except OSError:
        return entries
    return entries
