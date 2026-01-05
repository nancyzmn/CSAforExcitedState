from typing import List
from pathlib import Path
import fnmatch

def find_frame_dirs(root: Path, pattern: str = "frame*") -> List[Path]:
    """Return directories under root whose names match the given glob pattern, sorted numerically if possible."""
    dirs = [d for d in root.iterdir() if d.is_dir() and fnmatch.fnmatch(d.name, pattern)]

    def key_fn(p: Path):
        import re
        m = re.search(r'(\d+)', p.name)
        return (int(m.group(1)) if m else float('inf'), p.name)

    return sorted(dirs, key=key_fn)