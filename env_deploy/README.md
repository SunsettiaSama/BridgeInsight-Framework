# Environment Deployment Tools

This folder contains utilities for managing Python project dependencies and virtual environment setup.

## Overview

The deployment tools provide three main functions:
1. **Parse project imports** - Scan Python files and extract dependencies
2. **Export current environment** - Export installed packages to requirements.txt
3. **Batch install dependencies** - Install packages with error tracking and logging

## Files Description

### `parse_req.py`
Scans the entire project directory to automatically extract all third-party package imports.

**Configuration (top of file):**
- `PROJECT_DIR`: Project root directory (defaults to parent directory of env_deploy)
- `OUTPUT_FILE`: Path to generated requirements.txt
- `EXCLUDE_PACKAGES`: Set of package names to exclude from scanning
- `IMPORT_TO_PACKAGE`: Mapping of import names to actual package names (e.g., `cv2` → `opencv-python`)
- `EXCLUDE_DIRECTORIES`: Directories to skip during scanning (e.g., .venv, __pycache__)
- `DEFAULT_VERSION_SUFFIX`: Version suffix format for requirements (empty by default)

**Usage:**
```bash
python parse_req.py
```

**Output:**
- Generates `requirements.txt` in project root directory
- Outputs clean package list (one per line, no Chinese characters)
- Example format: `numpy`, `pandas==1.0.0`, `requests>=2.25.0`

**Example Console Output:**
```
Successfully generated F:\...\requirements.txt
Total dependencies: 12
  - beautifulsoup4
  - numpy
  - pandas
  - requests
```

---

### `get_env.py`
Exports the current Python environment's installed packages to requirements.txt, filtering out Anaconda/Conda specific packages.

**Configuration (top of file):**
- `EXCLUDED_PREFIXES`: Package name prefixes to exclude (e.g., `conda-`, `anaconda-`)
- `EXCLUDED_NAMES`: Exact package names to exclude (e.g., `spyder`, `navigator-updater`)

**Usage:**
```bash
python get_env.py
```

**Output:**
- Generates `requirements.txt` in env_deploy directory
- Freeze format: `package==version` (e.g., `numpy==1.21.0`)
- Filters out Anaconda-specific packages automatically

**Prerequisites:**
- `uv` package manager must be installed: `pip install uv`

**Example Console Output:**
```
===== Start exporting and filtering current environment dependencies =====
✅ Detected uv installed, version: 0.1.0
📥 Obtained 156 raw dependency packages from uv
🔍 Exclude Anaconda built-in package: spyder==5.1.5
📋 Total excluded 8 Anaconda built-in packages
✅ Filtered dependency file generated successfully! Path: F:\...\requirements.txt
📊 Final export 148 third-party dependency packages
```

---

### `install_env.py`
Batch installs packages from requirements.txt with comprehensive error tracking and logging.

**Configuration (top of file):**
- `REQUIREMENTS_PATH`: Path to requirements.txt file to install
- `FAILED_LOG_PATH`: Path for failed installation details log
- `RUN_LOG_PATH`: Path for complete run log
- `INSTALL_TIMEOUT`: Installation timeout per package (seconds, default: 300)

**Usage:**
```bash
python install_env.py
```

**Features:**
- Automatic encoding detection for requirements.txt file
- Supports UTF-8, UTF-16, GBK, GB2312, Latin-1 encodings
- Installs packages one-by-one to isolate failures
- Generates detailed failure logs for troubleshooting
- Creates run logs with timestamps

**Output Files:**
- `install_requirements_run.log`: Complete execution log
- `failed_install_packages.log`: Details of failed packages (if any)

**Prerequisites:**
- `uv` package manager must be installed: `pip install uv`
- `chardet` library for encoding detection: `pip install chardet`

**Example Console Output:**
```
2026-02-09 14:23:45 | INFO     | ===== Dependency Batch Installation Verification Script Started =====
2026-02-09 14:23:46 | INFO     | Detected uv installed, version: 0.1.0
2026-02-09 14:23:47 | INFO     | File encoding detected: utf-8 (confidence: 1.00)
2026-02-09 14:23:47 | INFO     | Successfully parsed requirements file, extracted 45 valid packages
2026-02-09 14:23:47 | INFO     | Starting installation of 45 dependency packages
2026-02-09 14:23:48 | INFO     | [1/45] Installing: numpy==1.21.0
2026-02-09 14:24:12 | INFO     | Installation successful: numpy==1.21.0
...
2026-02-09 14:25:30 | INFO     | ===== Installation Summary =====
2026-02-09 14:25:30 | INFO     | Total packages to install: 45
2026-02-09 14:25:30 | INFO     | Successfully installed: 44
2026-02-09 14:25:30 | INFO     | Failed installations: 1
```

**Log Example (failed_install_packages.log):**
```
===== Dependency Package Installation Failure Details Log =====
Generated at: 2026-02-09 14:25:30

Failed total: 1

Package: some-package==1.0.0
Error reason: No matching distribution found for some-package==1.0.0
--------------------------------------------------------------------------------
```

---

## Workflow Example

### Scenario 1: Export Dependencies from Current Environment

1. **Activate your Python environment** (where all your project dependencies are installed)
2. **Run get_env.py** to export current packages:
   ```bash
   python get_env.py
   ```
3. **Review requirements.txt** and adjust manually if needed

### Scenario 2: Parse Project Imports

1. **Update configuration** in parse_req.py if needed:
   - Modify `PROJECT_DIR` if projects are in different location
   - Add/remove excluded packages in `EXCLUDE_PACKAGES`
   - Update `IMPORT_TO_PACKAGE` mapping for custom imports
2. **Run parse_req.py**:
   ```bash
   python parse_req.py
   ```
3. **Review generated requirements.txt**

### Scenario 3: Fresh Environment Installation

1. **Ensure uv and chardet are installed**:
   ```bash
   pip install uv chardet
   ```
2. **Prepare requirements.txt** (using get_env.py or parse_req.py)
3. **Update REQUIREMENTS_PATH** in install_env.py if needed
4. **Run install_env.py**:
   ```bash
   python install_env.py
   ```
5. **Check logs** for any failed installations:
   - `install_requirements_run.log` - Full execution details
   - `failed_install_packages.log` - Only failed packages

---

## Troubleshooting

### Issue: "uv not installed"
**Solution:** Install uv package manager
```bash
pip install uv
```

### Issue: "chardet not installed" (in install_env.py)
**Solution:** Install chardet library
```bash
pip install chardet
```

### Issue: requirements.txt encoding error
**Solution:** install_env.py automatically detects and handles multiple encodings:
- UTF-8, UTF-16, GBK, GB2312, Latin-1
- Falls back to UTF-8 if all detection fails

### Issue: Some packages fail to install
**Solution:** Check `failed_install_packages.log` for details:
1. Verify package names are correct
2. Check network connectivity
3. Ensure package versions exist in PyPI
4. Try installing failed packages manually for more details

### Issue: parse_req.py returns "No dependencies detected"
**Solution:** Check configuration:
1. Verify `PROJECT_DIR` points to correct project root
2. Ensure Python files exist in PROJECT_DIR and subdirectories
3. Check that `EXCLUDE_DIRECTORIES` list doesn't exclude all code
4. Verify excluded packages list isn't too restrictive

---

## Dependencies

### Required
- Python 3.7+
- `uv` (0.1.0+)
- `chardet` (for install_env.py only)

### Installation
```bash
pip install uv chardet
```

---

## Notes

- All scripts use UTF-8 encoding for output files
- Timestamps in logs use format: `YYYY-MM-DD HH:MM:SS`
- Configuration constants are centralized at the top of each file for easy customization
- All Chinese characters have been removed for better cross-platform compatibility
