import subprocess
import logging
from pathlib import Path

logger = logging.getLogger("git_utils")
ROOT = Path(__file__).resolve().parent

def _run_git(args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

def _git_output(res: subprocess.CompletedProcess) -> str:
    return "\n".join(part for part in [res.stdout, res.stderr] if part).strip()

def _needs_pull_before_push(res: subprocess.CompletedProcess) -> bool:
    output = _git_output(res).lower()
    return any(
        hint in output
        for hint in [
            "fetch first",
            "non-fast-forward",
            "rejected",
            "failed to push some refs",
            "updates were rejected",
            "tip of your current branch is behind",
        ]
    )

def _pull_rebase_origin_main() -> bool:
    """Pull remote main while preserving locally-created commits."""
    res = _run_git(["pull", "--rebase", "origin", "main"], timeout=30)
    if res.returncode == 0:
        logger.info("Pulled latest origin/main with rebase before pushing.")
        return True
    logger.warning("Git pull --rebase origin main failed: %s", _git_output(res))
    return False

def _push_origin_main_with_retry() -> bool:
    """Push to origin/main, pulling remote commits first when Git asks for it."""
    push_res = _run_git(["push", "origin", "main"], timeout=30)
    if push_res.returncode == 0:
        return True

    if not _needs_pull_before_push(push_res):
        logger.warning("Git push failed: %s", _git_output(push_res))
        return False

    logger.info("Git push was rejected because origin/main is ahead; pulling first.")
    if not _pull_rebase_origin_main():
        return False

    retry_res = _run_git(["push", "origin", "main"], timeout=30)
    if retry_res.returncode == 0:
        logger.info("Successfully pushed changes after pulling origin/main.")
        return True

    logger.warning("Git push retry failed: %s", _git_output(retry_res))
    return False

def git_pull() -> bool:
    """Run git pull in the project root to fetch latest settings from cloud."""
    try:
        res = _run_git(["pull", "--rebase", "origin", "main"], timeout=30)
        if res.returncode == 0:
            logger.info("Successfully pulled latest changes from Git repository.")
            return True
        else:
            logger.warning("Git pull failed: %s", _git_output(res))
    except Exception as e:
        logger.exception("Failed to run git pull: %s", e)
    return False

def git_push_changes(message: str) -> bool:
    """Run git add, commit, and push in the project root for settings files."""
    try:
        # Check if there are changes to stage
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if status.returncode != 0:
            return False

        if not status.stdout.strip():
            # There may still be a local commit whose previous push failed.
            return _push_origin_main_with_retry()

        # Stage settings files
        files_to_stage = ["notification_config.json", "notification_state.json", "artifacts/favorites.json"]
        # Filter files that actually exist
        existing_files = [f for f in files_to_stage if (ROOT / f).exists()]
        if not existing_files:
            return True

        # Add files
        subprocess.run(["git", "add"] + existing_files, cwd=str(ROOT), check=True)
        # Commit
        commit_res = subprocess.run(["git", "commit", "-m", message], cwd=str(ROOT), capture_output=True, text=True)
        nothing_to_commit = (
            "nothing to commit" in commit_res.stdout
            or "nothing to commit" in commit_res.stderr
        )
        if commit_res.returncode != 0 and not nothing_to_commit:
            logger.warning("Git commit failed: %s", commit_res.stderr or commit_res.stdout)
            return False
        # Push
        if _push_origin_main_with_retry():
            logger.info("Successfully pushed changes to Git: %s", message)
            return True
    except Exception as e:
        logger.exception("Failed to run git push: %s", e)
    return False
