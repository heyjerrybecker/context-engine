import os
import sys
import tempfile
from pathlib import Path
import pytest

# Add scripts/lib to sys.path for metrics tests
_scripts = str(Path(__file__).parent.parent / "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)


@pytest.fixture
def tmp_db(tmp_path):
    """Return path to a temporary SQLite database."""
    return str(tmp_path / "test_graph.db")


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Return a temporary config directory with example config."""
    config_dir = tmp_path / ".context-engine"
    config_dir.mkdir()

    soul = tmp_path / "SOUL.md"
    soul.write_text("# Test Agent\nYou are a helpful test agent.")

    user = tmp_path / "USER.md"
    user.write_text("# Test User\nPrefers concise responses.")

    config = config_dir / "config.yaml"
    config.write_text(
        f"identity:\n"
        f"  soul: {soul}\n"
        f"  user_model: {user}\n"
        f"server:\n"
        f"  port: 8850\n"
        f"confidence:\n"
        f"  decay_half_life_days: 14\n"
        f"  archive_threshold: 0.1\n"
    )
    return str(config_dir)
