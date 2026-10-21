import json

import pytest

from crlsim.config_io import (
    append_history,
    load_configuration,
    load_history,
    save_configuration,
)


def test_configuration_round_trip(tmp_path):
    path = tmp_path / "beamline.json"
    configuration = {"mode": "2D", "energy": 12.0, "stacks": [{"material": "Be"}]}
    save_configuration(path, configuration)
    assert load_configuration(path) == configuration


def test_history_is_newest_first_and_deduplicated(tmp_path):
    path = tmp_path / "history.json"
    first = {"mode": "1D", "stacks": [{"material": "Be"}]}
    second = {"mode": "2D", "stacks": [{"material": "diamond"}]}
    append_history(first, path)
    append_history(first, path)
    entries = append_history(second, path)
    assert [entry["configuration"] for entry in entries] == [second, first]
    assert load_history(path) == entries


def test_invalid_schema_is_rejected(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"schema_version": 999, "configuration": {}}))
    with pytest.raises(ValueError, match="schema version"):
        load_configuration(path)


@pytest.mark.parametrize("document", [[], "bad", 1, None])
def test_non_object_configuration_is_rejected(tmp_path, document):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="JSON object"):
        load_configuration(path)


def test_corrupt_configuration_is_rejected_cleanly(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{")
    with pytest.raises(ValueError, match="Could not read"):
        load_configuration(path)


def test_history_filters_invalid_entries_deduplicates_and_truncates(tmp_path):
    path = tmp_path / "history.json"
    entries = [
        {"timestamp": f"t{index}", "configuration": {"value": index}}
        for index in range(35)
    ]
    entries.insert(1, entries[0].copy())
    entries.extend([None, {}, {"timestamp": 2, "configuration": {}}])
    path.write_text(json.dumps({"schema_version": 1, "history": entries}))
    loaded = load_history(path)
    assert len(loaded) == 30
    assert loaded[0]["configuration"] == {"value": 0}
    assert loaded[1]["configuration"] == {"value": 1}


@pytest.mark.parametrize("document", [[], "bad", 1, None, {"history": []}])
def test_invalid_history_document_returns_empty(tmp_path, document):
    path = tmp_path / "history.json"
    path.write_text(json.dumps(document))
    assert load_history(path) == []
