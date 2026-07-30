from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
import hashlib
import json
import subprocess

MANIFEST_VERSION = 1
_PACKAGE_NAME = "qmregion-selector"
_PACKAGE_DIR = Path(__file__).resolve().parent


def get_package_version() -> Optional[str]:
    """Installed qmregion-selector version, or None if not installed as a package."""
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return None


def get_git_info(repo_dir: Path = _PACKAGE_DIR) -> Tuple[Optional[str], Optional[bool]]:
    """
    (commit_sha, is_dirty) for the git checkout containing the qmregion_selector
    source, or (None, None) if it isn't a git checkout (e.g. installed from a
    tarball on an HPC system) or git isn't available. Best-effort only — this
    describes the code, not a guarantee of reproducibility.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir, capture_output=True, text=True, check=True, timeout=5,
        ).stdout
        return commit, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, None


def hash_atom_indices(atoms: Iterable[int]) -> str:
    """
    Stable SHA-256 over a sorted, de-duplicated atom-index list — lets two
    manifests be compared for an exact-region match without embedding the
    (potentially long) atom list itself.
    """
    canonical = ",".join(str(int(a)) for a in sorted(set(int(a) for a in atoms)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunManifest:
    """
    Provenance record for one QMRegionSelector run, so results stay
    comparable across systems: what code produced them (version, git commit),
    what was asked for (`config`, the resolved config dict verbatim), and
    what was actually observed/computed (`results`).
    """

    manifest_version: int
    generated_at: str
    qmregion_selector_version: Optional[str]
    git_commit: Optional[str]
    git_dirty: Optional[bool]
    config: Dict[str, Any]
    results: Dict[str, Any]

    @classmethod
    def build(cls, config: Dict[str, Any], results: Dict[str, Any]) -> "RunManifest":
        """Fill in the auto-derived fields (version, git, timestamp) around a config/results pair."""
        commit, dirty = get_git_info()
        return cls(
            manifest_version=MANIFEST_VERSION,
            generated_at=datetime.now(timezone.utc).isoformat(),
            qmregion_selector_version=get_package_version(),
            git_commit=commit,
            git_dirty=dirty,
            config=config,
            results=results,
        )

    def to_file(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_file(cls, path: Path) -> "RunManifest":
        data = json.loads(Path(path).read_text())
        return cls(**data)
