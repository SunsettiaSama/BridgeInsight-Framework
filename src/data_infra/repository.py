from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

from src.data_infra.models import MetadataRow, metadata_row_from_db_row, to_legacy_metadata_dict
from src.data_infra.mysql_client import MySQLClient

TimestampKey = Tuple[int, int, int]


class MetadataRepository:
    def __init__(self, client: MySQLClient) -> None:
        self._client = client

    @staticmethod
    def to_legacy_wind_dict(row: MetadataRow) -> dict:
        return to_legacy_metadata_dict(row)

    def get_wind_meta(
        self,
        dataset_tag: str,
        year: int,
        month: int,
        day: int,
        hour: int,
        sensor_id: str,
    ) -> Optional[MetadataRow]:
        row = self._client.fetchone(
            """
            SELECT dataset_tag, year, month, day, hour, sensor_id, file_path, raw_metadata_json
            FROM wind_metadata_hourly
            WHERE dataset_tag = %s AND year = %s AND month = %s AND day = %s AND hour = %s AND sensor_id = %s
            LIMIT 1
            """,
            (dataset_tag, int(year), int(month), int(day), int(hour), str(sensor_id)),
        )
        if row is None:
            return None
        return metadata_row_from_db_row(row)

    def get_wind_meta_legacy(
        self,
        dataset_tag: str,
        year: int,
        month: int,
        day: int,
        hour: int,
        sensor_id: str,
    ) -> Optional[dict]:
        row = self.get_wind_meta(dataset_tag, year, month, day, hour, sensor_id)
        if row is None:
            return None
        return self.to_legacy_wind_dict(row)

    def list_wind_meta_at_timestamp(
        self,
        dataset_tag: str,
        year: int,
        month: int,
        day: int,
        hour: int,
        sensor_ids: Optional[Sequence[str]] = None,
    ) -> List[MetadataRow]:
        params: List = [dataset_tag, int(year), int(month), int(day), int(hour)]
        sensor_filter = ""
        if sensor_ids:
            placeholders = ",".join(["%s"] * len(sensor_ids))
            sensor_filter = f" AND sensor_id IN ({placeholders})"
            params.extend(str(sid) for sid in sensor_ids)
        rows = self._client.fetchall(
            f"""
            SELECT dataset_tag, year, month, day, hour, sensor_id, file_path, raw_metadata_json
            FROM wind_metadata_hourly
            WHERE dataset_tag = %s AND year = %s AND month = %s AND day = %s AND hour = %s
            {sensor_filter}
            ORDER BY sensor_id
            """,
            params,
        )
        return [metadata_row_from_db_row(row) for row in rows]

    def resolve_wind_meta_with_priority(
        self,
        dataset_tag: str,
        year: int,
        month: int,
        day: int,
        hour: int,
        candidate_sensor_ids: Sequence[str],
    ) -> Optional[dict]:
        if not candidate_sensor_ids:
            return None
        for sensor_id in candidate_sensor_ids:
            legacy = self.get_wind_meta_legacy(
                dataset_tag, year, month, day, hour, str(sensor_id)
            )
            if legacy is not None:
                return legacy
        return None

    def get_vib_meta(
        self,
        dataset_tag: str,
        year: int,
        month: int,
        day: int,
        hour: int,
        sensor_id: str,
    ) -> Optional[MetadataRow]:
        row = self._client.fetchone(
            """
            SELECT dataset_tag, year, month, day, hour, sensor_id, file_path, raw_metadata_json
            FROM vib_metadata_hourly
            WHERE dataset_tag = %s AND year = %s AND month = %s AND day = %s AND hour = %s AND sensor_id = %s
            LIMIT 1
            """,
            (dataset_tag, int(year), int(month), int(day), int(hour), str(sensor_id)),
        )
        if row is None:
            return None
        return metadata_row_from_db_row(row)

    def get_wind_sensor_candidates(self, vib_sensor_id: str) -> List[str]:
        rows = self._client.fetchall(
            """
            SELECT wind_sensor_id, priority
            FROM vib_to_wind_mapping
            WHERE vib_sensor_id = %s
            ORDER BY priority ASC
            """,
            (str(vib_sensor_id),),
        )
        return [str(row["wind_sensor_id"]) for row in rows]

    def count_rows(self, table: str, dataset_tag: str) -> int:
        row = self._client.fetchone(
            f"SELECT COUNT(*) AS cnt FROM {table} WHERE dataset_tag = %s",
            (dataset_tag,),
        )
        return int(row["cnt"]) if row else 0

    def upsert_vib_rows(self, rows: Sequence[MetadataRow]) -> int:
        return self._upsert_rows("vib_metadata_hourly", rows)

    def upsert_wind_rows(self, rows: Sequence[MetadataRow]) -> int:
        return self._upsert_rows("wind_metadata_hourly", rows)

    def _upsert_rows(self, table: str, rows: Sequence[MetadataRow]) -> int:
        if not rows:
            return 0
        sql = f"""
            INSERT INTO {table} (
                dataset_tag, year, month, day, hour, sensor_id, file_path, raw_metadata_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                file_path = VALUES(file_path),
                raw_metadata_json = VALUES(raw_metadata_json),
                updated_at = CURRENT_TIMESTAMP
        """
        params = []
        for row in rows:
            raw_json = None
            if row.raw_metadata_json is not None:
                raw_json = json.dumps(row.raw_metadata_json, ensure_ascii=False)
            params.append(
                (
                    row.dataset_tag,
                    row.year,
                    row.month,
                    row.day,
                    row.hour,
                    row.sensor_id,
                    row.file_path,
                    raw_json,
                )
            )
        return self._client.executemany(sql, params)

    def seed_vib_to_wind_mapping(self, mapping: Dict[str, Sequence[str]]) -> int:
        rows = []
        for vib_sensor_id, wind_sensor_ids in mapping.items():
            for priority, wind_sensor_id in enumerate(wind_sensor_ids):
                rows.append((str(vib_sensor_id), str(wind_sensor_id), int(priority)))
        if not rows:
            return 0
        self._client.execute("DELETE FROM vib_to_wind_mapping")
        return self._client.executemany(
            """
            INSERT INTO vib_to_wind_mapping (vib_sensor_id, wind_sensor_id, priority)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE priority = VALUES(priority)
            """,
            rows,
        )
