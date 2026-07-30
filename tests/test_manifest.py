import json

from qmregion_selector.manifest import (
    MANIFEST_VERSION,
    RunManifest,
    get_git_info,
    get_package_version,
    hash_atom_indices,
)


def test_get_package_version_returns_installed_version():
    # This repo is installed editable in the test environment.
    assert get_package_version() == "0.1.0"


def test_get_git_info_on_this_repo_returns_a_commit_and_dirty_flag():
    commit, dirty = get_git_info()
    assert commit is not None
    assert len(commit) == 40  # full SHA-1 hex
    assert isinstance(dirty, bool)


def test_get_git_info_returns_none_outside_a_git_checkout(tmp_path):
    commit, dirty = get_git_info(repo_dir=tmp_path)
    assert commit is None
    assert dirty is None


def test_hash_atom_indices_is_order_independent_and_deduplicates():
    assert hash_atom_indices([3, 1, 2]) == hash_atom_indices([1, 2, 3, 2])


def test_hash_atom_indices_differs_for_different_sets():
    assert hash_atom_indices([1, 2, 3]) != hash_atom_indices([1, 2, 4])


def test_run_manifest_build_fills_in_derived_fields():
    manifest = RunManifest.build(config={"a": 1}, results={"b": 2})
    assert manifest.manifest_version == MANIFEST_VERSION
    assert manifest.qmregion_selector_version == "0.1.0"
    assert manifest.config == {"a": 1}
    assert manifest.results == {"b": 2}
    assert manifest.generated_at  # non-empty timestamp string


def test_run_manifest_round_trips_through_json(tmp_path):
    manifest = RunManifest.build(config={"topfile": "x.prmtop"}, results={"n_atoms": 10})
    path = tmp_path / "run_manifest.json"
    manifest.to_file(path)

    loaded = RunManifest.from_file(path)
    assert loaded == manifest

    # And it's plain, human-readable JSON, not something manifest-specific.
    raw = json.loads(path.read_text())
    assert raw["config"] == {"topfile": "x.prmtop"}
    assert raw["results"] == {"n_atoms": 10}
