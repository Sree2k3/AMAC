import os
import pytest
from backend.app.services.repo_processing import process_repository

@pytest.mark.asyncio
async def test_small_public_repo():
    # Use a tiny public repo that definitely fits our limits
    repo_url = 'https://github.com/psf/requests'   # ~2 MB, many .py files
    files, tmp_dir = process_repository(repo_url)   # token not needed for public repos

    # Basic sanity checks
    assert len(files) > 0, "No source files were found"
    for fp in files:
        # Every file must be a pathlib.Path and have a supported extension
        assert fp.suffix.lower() in {'.py', '.js', '.ts', '.tsx', '.java', '.go', '.c', '.cpp'}

    # Clean up
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

def test_repo_too_big(monkeypatch):
    # Simulate a repo that exceeds the size limit
    monkeypatch.setenv('MAX_REPO_SIZE_MB', '0')   # force limit to 0 MiB
    repo_url = 'https://github.com/psf/requests'
    with pytest.raises(ValueError) as exc:
        process_repository(repo_url)
    assert 'exceeds limit' in str(exc.value)
