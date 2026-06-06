"""Configuration for the Context Engine."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_CONFIG_DIR = os.path.expanduser("~/.context-engine")


@dataclass
class ContextEngineConfig:
    config_dir: str = DEFAULT_CONFIG_DIR
    soul_path: str = ""
    user_model_path: str = ""
    port: int = 8850
    host: str = "127.0.0.1"
    decay_half_life_days: float = 14
    archive_threshold: float = 0.1
    grace_days: float = 7
    redis_url: str = ""
    cache_default_timeout: int = 60
    extraction_model: str = "claude-haiku-4-5@20251001"

    @property
    def db_path(self) -> str:
        return os.path.join(self.config_dir, "graph.db")

    def load_soul(self) -> str:
        if self.soul_path and os.path.exists(self.soul_path):
            return Path(self.soul_path).read_text()
        return ""

    def load_user_model(self) -> str:
        if self.user_model_path and os.path.exists(self.user_model_path):
            return Path(self.user_model_path).read_text()
        return ""


def load_config(config_dir: str = DEFAULT_CONFIG_DIR) -> ContextEngineConfig:
    config_file = os.path.join(config_dir, "config.yaml")
    cfg = ContextEngineConfig(config_dir=config_dir)

    if os.path.exists(config_file):
        with open(config_file) as f:
            data = yaml.safe_load(f) or {}

        identity = data.get("identity", {})
        cfg.soul_path = str(identity.get("soul", ""))
        cfg.user_model_path = str(identity.get("user_model", ""))

        server = data.get("server", {})
        cfg.port = server.get("port", 8850)
        cfg.host = server.get("host", "127.0.0.1")

        confidence = data.get("confidence", {})
        cfg.decay_half_life_days = confidence.get("decay_half_life_days", 14)
        cfg.archive_threshold = confidence.get("archive_threshold", 0.1)
        cfg.grace_days = confidence.get("grace_days", 7)

        cache = data.get("cache", {})
        cfg.redis_url = cache.get("redis_url", "")
        cfg.cache_default_timeout = cache.get("default_timeout", 60)

        extraction = data.get("extraction", {})
        cfg.extraction_model = extraction.get("model", "claude-haiku-4-5@20251001")

    return cfg
