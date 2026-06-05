import pytest
from context_engine.config import ContextEngineConfig, load_config, DEFAULT_CONFIG_DIR


class TestConfig:
    def test_load_from_directory(self, tmp_config_dir):
        cfg = load_config(tmp_config_dir)
        assert cfg.port == 8850
        assert cfg.decay_half_life_days == 14
        assert cfg.archive_threshold == 0.1

    def test_load_soul_md(self, tmp_config_dir):
        cfg = load_config(tmp_config_dir)
        soul = cfg.load_soul()
        assert "Test Agent" in soul

    def test_load_user_model(self, tmp_config_dir):
        cfg = load_config(tmp_config_dir)
        user = cfg.load_user_model()
        assert "Test User" in user

    def test_missing_soul_returns_empty(self, tmp_config_dir):
        cfg = load_config(tmp_config_dir)
        cfg.soul_path = "/nonexistent/SOUL.md"
        assert cfg.load_soul() == ""

    def test_db_path(self, tmp_config_dir):
        cfg = load_config(tmp_config_dir)
        assert cfg.db_path.endswith("graph.db")

    def test_default_config_dir(self):
        assert DEFAULT_CONFIG_DIR.endswith(".context-engine")
