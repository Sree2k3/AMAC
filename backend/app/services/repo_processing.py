import os
import shutil
import tempfile
import pathlib
from typing import List

from git import Repo, GitCommandError

from .git import list_code_files
from .db import get, set   # tiny KV cache – optional usage in this file
from dotenv import load_dotenv

load_dotenv()  # ensure .env variables are loaded when the module is imported

# -------------------------------------------------------------------------
# Helper: compute the total size (in MB) of a directory tree.
# -------------------------------------------------------------------------
def _directory_size_mb(root: pathlib.Path) -> float:
    total_bytes = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            fp = pathlib.Path(dirpath) / fn
            try:
                total_bytes += fp.stat().st_size
            except OSError:
                # Permission errors etc. – just skip the file
                continue
    return total_bytes / (1024 * 1024)


# -------------------------------------------------------------------------
# Core function – clone, validate size, list code files.
# -------------------------------------------------------------------------
def process_repository(repo_url: str, github_token: str | None = None) -> List[pathlib.Path]:
    """
    Clone a public repo (or a private one if a token is supplied) into a
    temporary directory, enforce size limits, and return all qualifying source‑code
    files as pathlib.Path objects.

    Raises:
        ValueError – when the repo is too large or a file is too large.
        RuntimeError – on cloning failures.
    """
    # 1️⃣ Build the URL that includes the token (if any)
    clone_url = repo_url
    if github_token:
        # Insert token after the protocol part: https://TOKEN@github.com/owner/repo.git
        clone_url = repo_url.replace("https://", f"https://{github_token}@")

    # 2️⃣ Create a fresh temporary directory for the clone
    temp_dir = pathlib.Path(tempfile.mkdtemp(prefix="amac_repo_"))

    try:
        # 3️⃣ Perform the clone – we use a shallow clone (depth=1) to keep it fast
        Repo.clone_from(clone_url, temp_dir, depth=1, multi_options=["--quiet"])
    except GitCommandError as exc:
        # Clean up before bubbling the error up
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Git clone failed: {exc}") from exc

    # 4️⃣ Enforce total repository size limit
    max_repo_mb = int(os.getenv("MAX_REPO_SIZE_MB", "200"))
    repo_size_mb = _directory_size_mb(temp_dir)
    if repo_size_mb > max_repo_mb:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError(
            f"Repository size {repo_size_mb:.1f} MiB exceeds limit of {max_repo_mb} MiB."
        )

    # 5️⃣ List all source files (the helper lives in services/git.py)
    all_files = list_code_files(temp_dir)

    # 6️⃣ Filter out any file that is larger than the per‑file limit
    max_file_kb = int(os.getenv("MAX_FILE_SIZE_KB", "500"))
    filtered = []
    for fp in all_files:
        try:
            size_kb = fp.stat().st_size / 1024
            if size_kb <= max_file_kb:
                filtered.append(fp)
        except OSError:
            continue   # skip unreadable files

    # 7️⃣ Return the list – caller is responsible for cleaning the temp dir later
    return filtered, temp_dir   # we also hand the temp_dir back so the caller can clean it
