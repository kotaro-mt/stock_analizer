import subprocess
import logging
from pathlib import Path

logger = logging.getLogger("git_utils")
ROOT = Path(__file__).resolve().parent

def git_pull() -> bool:
    """Run git pull in the project root to fetch latest settings from cloud."""
    try:
        res = subprocess.run(
            ["git", "pull"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        if res.returncode == 0:
            logger.info("Successfully pulled latest changes from Git repository.")
            return True
        else:
            logger.warning("Git pull failed: %s", res.stderr or res.stdout)
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
            push_res = subprocess.run(
                ["git", "push"], cwd=str(ROOT), capture_output=True,
                text=True, timeout=30,
            )
            return push_res.returncode == 0

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
        push_res = subprocess.run(["git", "push"], cwd=str(ROOT), capture_output=True, text=True, timeout=30)
        if push_res.returncode == 0:
            logger.info("Successfully pushed changes to Git: %s", message)
            return True
        else:
            logger.warning("Git push failed: %s", push_res.stderr or push_res.stdout)
    except Exception as e:
        logger.exception("Failed to run git push: %s", e)
    return False
