# Data Processor Module Documentation

This module handles the entire lifecycle of vibration and wind data, from raw file unpacking to high-level database management and signal processing.

## 📂 File Structure & Responsibilities

### 1. `io_unpacker.py` (Low-Level I/O)
- **Role**: The foundational layer for data access.
- **Key Class**: `UNPACK`, `DataManager`.
- **Functions**:
    - `VIC_DATA_Unpack`: Decodes binary `.VIC` vibration data files.
    - `Wind_Data_Unpack`: Parses `.UAN` text-based wind data files.
    - `File_Read_Paths`: Recursively scans directories for data files.
    - `File_Detach_Data`: High-level interface to get data segments by time intervals.
- **Dependency**: Standard libraries (`struct`, `re`), `numpy`.

### 2. `algorithms.py` (Core Algorithms)
- **Role**: Contains pure mathematical and signal processing algorithms.
- **Key Functions**:
    - `isVIV`: Identifies Vortex-Induced Vibration (VIV) using Power Spectral Density (PSD) and peak-to-peak criteria.
    - `wind_turbulence_intensity_cal`: Calculates Turbulence Intensity (TI) based on IEC 61400-1 standards.
    - `apply_function`: A high-performance, multi-threaded framework for applying custom functions to Pandas DataFrames containing list-like data.
- **Dependency**: `numpy`, `scipy`.

### 3. `persistence_utils.py` (Data Utilities)
- **Role**: Helper functions for data cleaning and storage.
- **Key Functions**:
    - `save_dataframe_to_parquet` / `load_dataframe_from_parquet`: Native support for storing list-data in DataFrames using the Parquet format.
    - `clean_by_length`: Filters out data segments that don't match expected sampling lengths.
- **Dependency**: `pandas`, `pyarrow`.

### 4. `database_manager.py` (High-Level Data Management)
- **Role**: Manages a local "database" of Parquet chunks.
- **Key Class**: `ChunkManager`.
- **Features**:
    - **Multiprocessing Support**: Uses `multiprocessing.Manager` to share data and cache across process boundaries.
    - **Caching**: Implements a shared cache to avoid redundant calculations.
    - **Lazy Loading**: Only loads necessary data chunks based on sensor IDs and time ranges.
- **Dependency**: `multiprocessing`, `pandas`.

### 5. `pipeline_orchestrator.py` (Process Orchestration)
- **Role**: Orchestrates the flow from raw files to processed database chunks.
- **Key Class**: `TimeSeriesDataSegmenter`.
- **Functions**:
    - `process_df`: The main pipeline that performs:
        1. File parsing (via `io_unpacker`)
        2. Column splitting
        3. Time-window segmentation (e.g., splitting 1-hour files into 1-minute intervals).
    - `build_database_chunks`: Top-level function to build the entire sensor database from a configuration file.
- **Dependency**: `io_unpacker`, `persistence_utils`, `concurrent.futures`.

### 6. `statistics/` (Analytical Statistics)
- **`rms_statistics.py`**: Calculates global RMS quantiles (P95 and Top 0.25%) to identify extreme vibration events. Enriches results with time and sensor metadata.

## 🔄 Module Coupling & Architecture

The module follows a layered architecture:

1. **Pipeline Orchestrator** (Top) -> Calls **IO Unpacker** & **Persistence Utils**.
2. **Database Manager** (Mid) -> Manages the outputs of the Orchestrator.
3. **Algorithms** (Side) -> Used by both Orchestrator and external scripts for data analysis.
4. **IO Unpacker** (Base) -> Independent low-level file access.

## 🚀 Usage Example

### Building the Database
```python
from src.data_processer.pipeline_orchestrator import build_database_chunks

# Build 1-minute interval database from config.yaml
saved_files, config_path = build_database_chunks("config.yaml", interval_nums=60)
```

### Accessing Data via Manager
```python
from src.data_processer.database_manager import ChunkManager

manager = ChunkManager(local_dir="path/to/db")
# Data is automatically loaded and cached as needed
df = manager.get_chunk("sensor_group_A")
```

### Performing VIV Detection
```python
from src.data_processer.algorithms import isVIV
import numpy as np

vibration_data = np.random.randn(3000) # 1 minute of 50Hz data
is_viv = isVIV(vibration_data, f0=0.25)
```
