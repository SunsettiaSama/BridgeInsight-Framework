from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ENV_PATH = _PROJECT_ROOT / "src" / "data_infra" / "docker" / ".env"


@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str

    @property
    def dsn_kwargs(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
            "charset": "utf8mb4",
        }


def _load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_mysql_config(cfg: Optional[dict] = None) -> MySQLConfig:
    cfg = cfg or {}
    mysql_cfg = cfg.get("mysql") or {}
    dotenv = _load_dotenv(_DEFAULT_ENV_PATH)

    def pick(key: str, default: str) -> str:
        env_key = f"MYSQL_{key.upper()}"
        if env_key in os.environ:
            return os.environ[env_key]
        if key in mysql_cfg and mysql_cfg[key] is not None:
            return str(mysql_cfg[key])
        if key in dotenv:
            return dotenv[key]
        return default

    return MySQLConfig(
        host=pick("host", "127.0.0.1"),
        port=int(pick("port", "3306")),
        user=pick("user", "vibration"),
        password=pick("password", "vibration"),
        database=pick("database", "vibration_data"),
    )


def load_dataset_yaml(path: str | Path) -> dict:
    yaml_path = Path(path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


DATASET_REGISTRY: dict[str, dict] = {
    "2023": {
        "dataset_tag": "2023",
        "year": 2023,
        "yaml_path": _PROJECT_ROOT / "config" / "datasets" / "total_staycable_vib.yaml",
    },
    "202409": {
        "dataset_tag": "202409",
        "year": 2024,
        "yaml_path": _PROJECT_ROOT / "config" / "datasets" / "total_staycable_vib_202409.yaml",
    },
}
