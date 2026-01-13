from typing import Any, cast

import pytest

from capybara import PowerDict


def test_powerdict_init_accepts_none_and_kwargs():
    assert PowerDict() == {}
    assert PowerDict(None) == {}

    pd = PowerDict(None, new="value")
    assert pd == {"new": "value"}
    assert pd.new == "value"

    pd = PowerDict({"a": 1}, b=2)
    assert pd == {"a": 1, "b": 2}
    assert pd.a == 1
    assert pd.b == 2


def test_powerdict_attribute_and_item_access_are_kept_in_sync():
    pd = PowerDict()
    pd.alpha = 1
    assert pd["alpha"] == 1

    pd["beta"] = 2
    assert pd.beta == 2

    del pd["alpha"]
    assert "alpha" not in pd
    assert not hasattr(pd, "alpha")

    pd.gamma = 3
    del pd.gamma
    assert "gamma" not in pd


def test_powerdict_recursively_wraps_nested_mappings_and_sequences():
    pd = PowerDict(
        {
            "cfg": {"x": 1},
            "items": [{"y": 2}, {"z": 3}],
            "numbers_tuple": (1, 2, 3),
        }
    )

    assert isinstance(pd.cfg, PowerDict)
    assert pd.cfg.x == 1

    assert isinstance(pd.items, list)
    assert [type(x) for x in pd.items] == [PowerDict, PowerDict]
    assert pd.items[0].y == 2
    assert pd.items[1].z == 3

    # Tuples are normalized to lists to keep internal mutation rules simple.
    assert pd.numbers_tuple == [1, 2, 3]


def test_powerdict_freeze_blocks_mutation_and_melt_restores():
    pd = PowerDict({"a": 1, "nested": {"b": 2}})
    pd.freeze()

    with pytest.raises(ValueError, match="PowerDict is frozen"):
        pd.a = 10
    with pytest.raises(ValueError, match="PowerDict is frozen"):
        pd["a"] = 10
    with pytest.raises(ValueError, match="PowerDict is frozen"):
        del pd.a
    with pytest.raises(ValueError, match="PowerDict is frozen"):
        del pd["a"]
    with pytest.raises(ValueError, match="PowerDict is frozen"):
        pd.update({"c": 3})
    with pytest.raises(ValueError, match="PowerDict is frozen"):
        pd.pop("a")

    # Nested PowerDict is also frozen.
    with pytest.raises(ValueError, match="PowerDict is frozen"):
        pd.nested.b = 20

    pd.melt()
    pd.a = 10
    pd.update({"c": 3})
    assert pd == {"a": 10, "nested": {"b": 2}, "c": 3}


def test_powerdict_pop_behaves_like_dict_pop():
    pd = PowerDict({"a": 1})
    assert pd.pop("a") == 1
    assert "a" not in pd

    assert pd.pop("missing", None) is None

    with pytest.raises(KeyError):
        pd.pop("missing")


def test_powerdict_reserved_frozen_key_behaviors():
    pd = PowerDict()
    with pytest.raises(KeyError, match="_frozen"):
        pd["_frozen"] = True
    with pytest.raises(KeyError, match="_frozen"):
        del pd["_frozen"]
    with pytest.raises(KeyError, match="_frozen"):
        del pd._frozen


def test_powerdict_missing_attr_raises_attribute_error():
    pd = PowerDict()
    with pytest.raises(AttributeError):
        _ = pd.missing


def test_powerdict_serialization_helpers(tmp_path):
    pd = PowerDict({"a": 1, "nested": {"b": 2}})

    json_path = tmp_path / "x.json"
    yaml_path = tmp_path / "x.yaml"
    pkl_path = tmp_path / "x.pkl"
    txt_path = tmp_path / "x.txt"

    assert pd.to_json(json_path) is None
    assert json_path.exists()
    assert PowerDict.load_json(json_path) == pd

    assert pd.to_yaml(yaml_path) is None
    assert yaml_path.exists()
    assert PowerDict.load_yaml(yaml_path) == pd

    assert pd.to_pickle(pkl_path) is None
    assert pkl_path.exists()
    assert PowerDict.load_pickle(pkl_path) == pd

    pd.to_txt(txt_path)
    assert "nested" in txt_path.read_text(encoding="utf-8")


def test_powerdict_deepcopy_is_blocked_when_frozen():
    from copy import deepcopy

    pd = PowerDict({"a": 1})
    pd.freeze()
    with pytest.raises(Warning, match="cannot be copy"):
        _ = deepcopy(pd)


def test_powerdict_update_without_args_and_deepcopy_when_unfrozen():
    from copy import deepcopy

    pd = PowerDict({"a": 1})
    pd.update()
    assert pd == {"a": 1}

    clone = deepcopy(pd)
    assert clone == pd
    assert clone is not pd


def test_powerdict_freeze_and_to_dict_handle_powerdict_inside_lists():
    pd = PowerDict({"items": [{"x": 1}, {"y": 2}]})
    pd.freeze()
    with pytest.raises(ValueError, match="PowerDict is frozen"):
        items = cast(list[Any], pd["items"])
        cast(Any, items[0]).x = 10

    pd.melt()
    items = cast(list[Any], pd["items"])
    cast(Any, items[0]).x = 10
    assert cast(Any, items[0]).x == 10

    out = pd.to_dict()
    assert out == {"items": [{"x": 10}, {"y": 2}]}
