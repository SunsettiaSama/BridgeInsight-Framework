from __future__ import annotations

import gzip
import subprocess
import time
from pathlib import Path
from typing import Sequence

from src.data_infra.config import DATASET_REGISTRY, MySQLConfig, load_mysql_config
from src.data_infra.ingest import ingest_dataset
from src.data_infra.mysql_client import MySQLClient
from src.data_infra.repository import MetadataRepository

_DOCKER_DIR = Path(__file__).resolve().parent / "docker"
_COMPOSE_FILE = _DOCKER_DIR / "docker-compose.yml"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOT_DIR = _PROJECT_ROOT / "results" / "data_infra" / "mysql_snapshot"
_SNAPSHOT_TABLES = (
    "vib_metadata_hourly",
    "wind_metadata_hourly",
    "vib_to_wind_mapping",
)


def ensure_docker_mysql(max_wait_s: int = 120) -> None:
    env_file = _DOCKER_DIR / ".env"
    if not env_file.exists():
        example = _DOCKER_DIR / ".env.example"
        env_file.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")

    subprocess.run(
        ["docker", "compose", "-f", str(_COMPOSE_FILE), "up", "-d"],
        cwd=str(_DOCKER_DIR),
        check=True,
        encoding="utf-8",
        errors="replace",
    )

    mysql_cfg = load_mysql_config()
    client = MySQLClient(mysql_cfg)
    deadline = time.monotonic() + max_wait_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        time.sleep(3)
        try:
            client.ping()
            return
        except Exception as exc:
            last_error = exc
    detail = f": {last_error}" if last_error else ""
    raise RuntimeError(f"MySQL 未在 {max_wait_s}s 内就绪{detail}")


def _mysql_user_args(mysql_cfg: MySQLConfig) -> list[str]:
    return [f"-u{mysql_cfg.user}", f"-p{mysql_cfg.password}"]


def _snapshot_path(cfg: dict, dataset_key: str) -> Path:
    root = Path(str(cfg.get("mysql_snapshot_dir", _SNAPSHOT_DIR)))
    return root / f"{dataset_key}.sql.gz"


def _dataset_counts(client: MySQLClient, dataset_tag: str) -> dict[str, int]:
    vib = client.fetchone(
        "SELECT COUNT(*) AS cnt FROM vib_metadata_hourly WHERE dataset_tag = %s",
        (dataset_tag,),
    )
    wind = client.fetchone(
        "SELECT COUNT(*) AS cnt FROM wind_metadata_hourly WHERE dataset_tag = %s",
        (dataset_tag,),
    )
    mapping = client.fetchone("SELECT COUNT(*) AS cnt FROM vib_to_wind_mapping")
    return {
        "vib": int(vib["cnt"]) if vib else 0,
        "wind": int(wind["cnt"]) if wind else 0,
        "mapping": int(mapping["cnt"]) if mapping else 0,
    }


def _dataset_ready(client: MySQLClient, dataset_key: str) -> bool:
    dataset_tag = str(DATASET_REGISTRY[dataset_key]["dataset_tag"])
    counts = _dataset_counts(client, dataset_tag)
    return counts["vib"] > 0 and counts["wind"] > 0 and counts["mapping"] > 0


def _restore_snapshot(snapshot_path: Path, mysql_cfg: MySQLConfig) -> None:
    payload = gzip.decompress(snapshot_path.read_bytes())
    subprocess.run(
        ["docker", "exec", "-i", "vibration-mysql", "mysql", *_mysql_user_args(mysql_cfg), mysql_cfg.database],
        input=payload,
        check=True,
    )


def _write_snapshot(snapshot_path: Path, mysql_cfg: MySQLConfig) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    dump = subprocess.run(
        [
            "docker",
            "exec",
            "vibration-mysql",
            "mysqldump",
            *_mysql_user_args(mysql_cfg),
            "--single-transaction",
            "--default-character-set=utf8mb4",
            mysql_cfg.database,
            *_SNAPSHOT_TABLES,
        ],
        check=True,
        capture_output=True,
    ).stdout
    snapshot_path.write_bytes(gzip.compress(dump, compresslevel=6))


def ensure_mysql_metadata_cache(
    cfg: dict,
    dataset_keys: Sequence[str] | None = None,
) -> None:
    if not bool(cfg.get("mysql_metadata_cache_enabled", True)):
        return
    keys = list(dataset_keys or cfg.get("mysql_metadata_cache_datasets", ["202409"]))
    if not keys:
        return

    mysql_cfg = load_mysql_config(cfg)
    client = MySQLClient(mysql_cfg)
    client.ping()
    repository = MetadataRepository(client)
    batch_size = int(cfg.get("mysql_metadata_cache_ingest_batch_size", 5000))

    for dataset_key in keys:
        if dataset_key not in DATASET_REGISTRY:
            raise ValueError(f"未知数据集 key: {dataset_key}")
        snapshot = _snapshot_path(cfg, dataset_key)
        if _dataset_ready(client, dataset_key):
            if bool(cfg.get("mysql_snapshot_write_when_ready", True)) and not snapshot.exists():
                print(f"Writing MySQL metadata snapshot: {snapshot}")
                _write_snapshot(snapshot, mysql_cfg)
            continue
        if snapshot.exists():
            print(f"Restoring MySQL metadata snapshot: {snapshot}")
            _restore_snapshot(snapshot, mysql_cfg)
            if _dataset_ready(client, dataset_key):
                continue
            raise RuntimeError(f"MySQL metadata snapshot restored but dataset is incomplete: {dataset_key}")

        print(f"Building MySQL metadata cache from JSON: {dataset_key}")
        ingest_dataset(repository, dataset_key, batch_size=batch_size)
        if not _dataset_ready(client, dataset_key):
            raise RuntimeError(f"MySQL metadata ingest finished but dataset is incomplete: {dataset_key}")
        print(f"Writing MySQL metadata snapshot: {snapshot}")
        _write_snapshot(snapshot, mysql_cfg)


if __name__ == "__main__":
    ensure_docker_mysql()
    ensure_mysql_metadata_cache({})
    print("MySQL is ready")
