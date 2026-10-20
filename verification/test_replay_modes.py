"""Replay defaults and selection resolution before any hardware I/O."""
from datetime import datetime
import os

import pytest

from obstacle_validation import lerobot_replay as replay


@pytest.mark.parametrize("controller", ["cbf", "mcbf"])
@pytest.mark.parametrize("physical", [False, True])
def test_avoid_requires_available_annotations_before_loading_dataset_or_hardware(tmp_path, monkeypatch, controller, physical):
    monkeypatch.setattr(replay, "resolve_repo_path", lambda _: tmp_path / "manual_obstacles")
    monkeypatch.setattr(replay, "load_project_config", lambda *a: pytest.fail("loaded config before mode validation"))
    flags = ["replay", "--dataset", str(tmp_path/"missing"), "--output", str(tmp_path/"out"),
             "--mode", "avoid", "--avoidance-controller", controller]
    flags.append("--execute" if physical else "--no-execute")
    with pytest.raises(ValueError, match="No annotations.json found"):
        replay.run_replay(replay.parser().parse_args(flags))
    assert not (tmp_path/"out").exists()


def test_replay_defaults_to_execution_and_timestamped_output():
    args = replay.parser().parse_args(["replay", "--dataset", "dataset", "--mode", "avoid"])
    assert args.execute and args.execute_ack == replay.EXECUTE_ACK
    assert args.output.parent.as_posix() == "outputs/obstacle_validation"
    datetime.strptime(args.output.name, "annotation_replay_%Y%m%d_%H%M%S_%f%z")
    virtual = replay.parser().parse_args(["replay", "--dataset", "dataset", "--no-execute"])
    assert not virtual.execute


def test_latest_annotations_uses_capture_directory_time_and_skips_incomplete_captures(tmp_path, monkeypatch):
    monkeypatch.setattr(replay, "resolve_repo_path", lambda _: tmp_path)
    # Directory names can be old UUIDs or timestamps; file edit time must not
    # make an older capture override a newer one.
    old = tmp_path / "5384c2a5fab847e0aa99a68993f5a17b" / "annotations.json"
    new = tmp_path / "20261020_213129_067358+0800" / "annotations.json"
    for path in (old, new):
        path.parent.mkdir()
        path.write_text("{}")
    os.utime(old.parent, ns=(1, 1))
    os.utime(new.parent, ns=(2, 2))
    (tmp_path / "incomplete_capture").mkdir()
    assert replay.latest_manual_annotations() == new


def test_baseline_rejects_selection_instead_of_silently_ignoring_it(tmp_path):
    args = replay.parser().parse_args([
        "replay", "--dataset", str(tmp_path/"missing"), "--output", str(tmp_path/"out"),
        "--mode", "baseline", "--manual-selection", str(tmp_path/"selection.json")])
    with pytest.raises(ValueError, match="only used with --mode avoid"):
        replay.run_replay(args)
