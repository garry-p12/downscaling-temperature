"""Config loading: inheritance, interpolation, env override, and the invariants
the shipped configs have to satisfy for a checkpoint to remain loadable.
"""
from __future__ import annotations

import pytest
import yaml

from common.config import CONFIG_DIR, Cfg, deep_merge, load_config

DOMAIN_CONFIGS = ["data", "data_austin", "data_southcentral", "data_synthetic"]


# --------------------------------------------------------------------------- #
# Mechanism
# --------------------------------------------------------------------------- #
def test_deep_merge_recurses_and_child_wins():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    over = {"a": {"y": 20, "z": 30}, "c": 4}
    assert deep_merge(base, over) == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3, "c": 4}


def test_deep_merge_replaces_lists_rather_than_appending():
    # Channel order is a contract; appending would silently corrupt it.
    merged = deep_merge({"ch": ["a", "b", "c"]}, {"ch": ["a", "b"]})
    assert merged["ch"] == ["a", "b"]


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"x": 1}}
    deep_merge(base, {"a": {"x": 2}})
    assert base == {"a": {"x": 1}}


def test_extends_pulls_in_base_keys(tmp_path):
    (tmp_path / "parent.yaml").write_text("a: 1\nnested: {k: base, keep: yes}\n")
    child = tmp_path / "child.yaml"
    child.write_text("extends: parent.yaml\nnested: {k: child}\n")
    cfg = load_config(str(child))
    assert cfg["a"] == 1
    assert cfg["nested"] == {"k": "child", "keep": True}
    assert "extends" not in cfg


def test_interpolation_resolves_top_level_scalars(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text('root: "store_x"\nsources: {a: {dir: "${root}/raw/a"}}\n')
    assert load_config(str(p))["sources"]["a"]["dir"] == "store_x/raw/a"


def test_interpolation_of_unknown_key_raises(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text('a: "${nope}"\n')
    with pytest.raises(KeyError):
        load_config(str(p))


def test_extends_cycle_raises(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(RecursionError):
        load_config(str(tmp_path / "a.yaml"))


def test_env_override_redirects_a_stem(tmp_path, monkeypatch):
    p = tmp_path / "alt.yaml"
    p.write_text("domain: {name: elsewhere}\n")
    monkeypatch.setenv("DOWNSCALE_CONFIG_data", str(p))
    assert load_config("data")["domain"]["name"] == "elsewhere"


def test_cfg_attribute_access_is_recursive():
    cfg = Cfg({"a": {"b": {"c": 1}}})
    assert cfg.a.b.c == 1
    with pytest.raises(AttributeError):
        cfg.missing


# --------------------------------------------------------------------------- #
# The shipped configs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", DOMAIN_CONFIGS)
def test_domain_config_is_complete_after_merge(name):
    cfg = load_config(f"configs/{name}.yaml")
    for key in ("domain", "grid", "time", "sources", "dataset", "truth_source"):
        assert key in cfg, f"{name} lost '{key}' in the merge"
    for key in ("lon_min", "lon_max", "lat_min", "lat_max", "name"):
        assert key in cfg["domain"]
    assert cfg["domain"]["lon_min"] < cfg["domain"]["lon_max"]
    assert cfg["domain"]["lat_min"] < cfg["domain"]["lat_max"]


@pytest.mark.parametrize("name", DOMAIN_CONFIGS)
def test_no_unresolved_interpolation_markers(name):
    """A typo'd ${...} must not survive into a path that gets written to."""
    def walk(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from walk(v)
        elif isinstance(node, list):
            for v in node:
                yield from walk(v)
        elif isinstance(node, str):
            yield node

    for s in walk(load_config(f"configs/{name}.yaml")):
        assert "${" not in s


@pytest.mark.parametrize("name", DOMAIN_CONFIGS)
def test_store_paths_live_under_the_domain_store_root(name):
    cfg = load_config(f"configs/{name}.yaml")
    root = cfg["store_root"]
    assert cfg["dataset"]["out_zarr"].startswith(root + "/")
    for src, block in cfg["sources"].items():
        if src == "coastline":       # one global download, shared by all domains
            continue
        if "out_dir" in block:
            assert block["out_dir"].startswith(root + "/"), src


@pytest.mark.parametrize("name", DOMAIN_CONFIGS)
def test_domain_files_hold_only_deltas(name):
    """Guards the whole point of data_base.yaml: no re-pasted shared blocks."""
    raw = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())
    assert raw["extends"] == "data_base"
    for shared in ("grid", "time", "sources", "truth_source"):
        assert shared not in raw, (
            f"{name}.yaml redefines '{shared}' — it is inherited from "
            f"data_base.yaml and must not be duplicated")


@pytest.mark.parametrize("name", ["data_southcentral", "data_synthetic"])
def test_input_channel_count_matches_model_in_channels(name):
    """A domain the shipped model config is meant to run against must agree on
    channel count, or the first conv silently loads at the wrong input width.
    Includes the synthetic domain: the quickstart is the first thing a new
    clone runs, and it used to fail here."""
    model = load_config("model")
    cfg = load_config(f"configs/{name}.yaml")
    assert len(cfg["dataset"]["input_channels"]) == model["in_channels"]


def test_synthetic_domain_cannot_overwrite_a_real_store():
    """The quickstart writes with mode='w'. Its store root must not collide
    with a domain that holds hours of downloads."""
    synth = load_config("configs/data_synthetic.yaml")["store_root"]
    others = {load_config(f"configs/{n}.yaml")["store_root"]
              for n in DOMAIN_CONFIGS if n != "data_synthetic"}
    assert synth not in others


def test_temporal_splits_do_not_overlap():
    cfg = load_config("configs/data_southcentral.yaml")
    t = cfg["time"]
    assert t["train"][1] < t["val"][0] < t["val"][1] < t["test"][0]


def test_holdout_box_sits_inside_the_training_domain():
    cfg = load_config("configs/data_southcentral.yaml")
    dom, hold = cfg["domain"], cfg["holdout"]
    assert dom["lon_min"] < hold["lon_min"] < hold["lon_max"] < dom["lon_max"]
    assert dom["lat_min"] < hold["lat_min"] < hold["lat_max"] < dom["lat_max"]
