import os
import shutil
import tempfile
from git import Repo
from pathlib import Path
from typing import List

def clone_repo(repo_url: str, token: str = None) -> Path:
    """Clone a public GitHub repo to a temporary directory."""
    dest = Path(tempfile.mkdtemp())
    if token:
        repo_url = repo_url.replace(
            "https://", f"https://{token}@"
        )
    Repo.clone_from(repo_url, dest)
    return dest

def list_code_files(
    root: Path,
    extensions=(".py", ".js", ".ts", ".tsx", ".java", ".go", ".c", ".cpp"),
    skip_dirs=("node_modules", "vendor", "__pycache__", ".git"),
) -> List[Path]:
    """Return an absolute list of all supported source-code files."""
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            if fn.lower().endswith(extensions):
                files.append(Path(dirpath) / fn)
    return files

