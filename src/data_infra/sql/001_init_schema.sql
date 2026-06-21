CREATE DATABASE IF NOT EXISTS vibration_data
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE vibration_data;

CREATE TABLE IF NOT EXISTS vib_metadata_hourly (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  dataset_tag VARCHAR(16) NOT NULL,
  year SMALLINT UNSIGNED NOT NULL,
  month TINYINT UNSIGNED NOT NULL,
  day TINYINT UNSIGNED NOT NULL,
  hour TINYINT UNSIGNED NOT NULL,
  sensor_id VARCHAR(64) NOT NULL,
  file_path TEXT NOT NULL,
  raw_metadata_json JSON NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_vib_hourly (dataset_tag, year, month, day, hour, sensor_id),
  KEY idx_vib_ts (dataset_tag, year, month, day, hour),
  KEY idx_vib_sensor_ts (dataset_tag, sensor_id, year, month, day, hour)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS wind_metadata_hourly (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  dataset_tag VARCHAR(16) NOT NULL,
  year SMALLINT UNSIGNED NOT NULL,
  month TINYINT UNSIGNED NOT NULL,
  day TINYINT UNSIGNED NOT NULL,
  hour TINYINT UNSIGNED NOT NULL,
  sensor_id VARCHAR(64) NOT NULL,
  file_path TEXT NOT NULL,
  raw_metadata_json JSON NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_wind_hourly (dataset_tag, year, month, day, hour, sensor_id),
  KEY idx_wind_ts (dataset_tag, year, month, day, hour),
  KEY idx_wind_sensor_ts (dataset_tag, sensor_id, year, month, day, hour)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS vib_to_wind_mapping (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  vib_sensor_id VARCHAR(64) NOT NULL,
  wind_sensor_id VARCHAR(64) NOT NULL,
  priority TINYINT UNSIGNED NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_vib_wind (vib_sensor_id, wind_sensor_id),
  KEY idx_vib_sensor (vib_sensor_id, priority)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
