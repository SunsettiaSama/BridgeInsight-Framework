# RMS Statistics Module Documentation

This module provides tools for calculating and analyzing the Root Mean Square (RMS) values of vibration signals, with a focus on identifying extreme vibration events using statistical quantiles.

## 📂 File Location
`src/data_processer/statistics/rms_statistics.py`

## 🛠️ Key Features

### 1. Statistical Threshold Calculation
- **Primary Threshold (P95)**: Calculates the 95th percentile of all RMS samples across the dataset.
- **Secondary Threshold (Extreme)**: Calculates the 95th percentile of the samples that already exceed the primary threshold (effectively the top 0.25% of all samples). This aligns with the "extreme vibration" logic used in thesis figures.

### 2. Parallel Processing
- Uses `ProcessPoolExecutor` to compute RMS values for thousands of `.VIC` files in parallel, significantly reducing processing time.

### 3. Metadata Enrichment
- Automatically enriches results with path-based metadata using `io_unpacker.parse_path_metadata`.
- Each record in the output JSON includes:
    - `path`: Original file path.
    - `indices`: List of window starting indices where RMS exceeds the extreme threshold.
    - `data_type`: Extracted from path (e.g., "VIC").
    - `month`, `day`: Time components extracted from the directory structure.
    - `sensor_id`: Extracted from the filename.
    - `hour`, `minute`, `second`: Accurate time components extracted from the filename suffix.

### 4. Detailed Reporting
- Generates a comprehensive terminal report including:
    - Threshold values ($m/s^2$).
    - Sample distribution (counts and percentages).
    - File-level statistics (e.g., number of files containing extreme vibrations).

## 🚀 Usage

### Running the Statistics Pipeline
To execute the full calculation and save results to JSON:
```bash
python src/data_processer/statistics/rms_statistics.py
```

### Loading Metadata in Other Scripts
```python
from src.data_processer.statistics.rms_statistics import load_rms_metadata

# Load the enriched metadata from the default JSON path
metadata = load_rms_metadata()

for entry in metadata:
    print(f"Sensor {entry['sensor_id']} had extreme vibration at {entry['hour']}:{entry['minute']}")
```

## 📊 Output
The results are saved to:
`results/statistics/rms_statistics.json`

Format:
```json
[
    {
        "path": "F:\\...\\ST-VIC-C18-102-01_130000.VIC",
        "indices": [3000, 6000],
        "data_type": "VIC",
        "month": "09",
        "day": "11",
        "sensor_id": "ST-VIC-C18-102-01",
        "hour": "13",
        "minute": "00",
        "second": "00",
        "raw_time": "130000"
    }
]
```
