# 环境部署指南

## 📋 概述

本目录包含了通过 `uv` 快速部署 Python 环境的完整工具：
- `deploy_env.sh`：自动化部署脚本
- `requirements.txt`：完整依赖清单（含本地路径）
- `requirements_core.txt`：核心依赖清单（仅 PyPI 可用）

## 🔧 使用方法

### 1. 快速开始（推荐）

```bash
cd env_deploy

# 编辑配置（修改目标部署目录等）
nano deploy_env.sh

# 运行部署脚本
bash deploy_env.sh
```

### 2. 配置说明

编辑 `deploy_env.sh` 中的【用户配置区】：

```bash
# 1. 目标部署目录
PROJECT_DIR="/opt/my_vibration_project"

# 2. 本地项目源代码路径（可选，留空则只部署环境）
SOURCE_PROJECT_DIR="/path/to/your/local/project"

# 3. requirements.txt 路径（默认使用脚本所在目录）
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

# 4. 是否使用锁文件（推荐设为 0）
USE_LOCK_FILE=0

# 5. Python 版本（留空使用默认）
PYTHON_VERSION="3.11"

# 6. 是否从本地环境克隆（可选）
USE_LOCAL_ENV=0
LOCAL_ENV_PATH=""
```

### 3. 脚本功能

部署脚本会自动完成以下步骤：

1. **清理依赖清单**：自动移除 `@ file:///...` 本地路径依赖
2. **安装 uv**：检查并安装 uv 包管理器
3. **验证本地环境**（可选）：如果启用克隆模式
4. **创建虚拟环境**：使用 uv 创建高速虚拟环境
5. **安装依赖**：从清理后的 requirements.txt 安装
6. **复制项目代码**（可选）：复制源代码到目标目录
7. **验证环境**：检查 Python 版本和已安装包

## ⚠️ 注意事项

### requirements.txt 清理

原始 `requirements.txt` 包含 **436** 个依赖，其中约 **70%** 是从本地文件安装的：
```
@ file:///C:/b/abs_xxx/...
@ file:///tmp/build/...
```

这些依赖**无法在新环境中使用**，脚本会自动清理并生成 `requirements_cleaned.txt`。

### 推荐使用 requirements_core.txt

如果完整的 requirements.txt 安装失败，建议使用精简的核心依赖：

```bash
# 修改配置，使用核心依赖文件
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements_core.txt"
```

`requirements_core.txt` 仅包含约 **40** 个核心包，均可从 PyPI 直接安装。

### 锁文件使用建议

**不推荐启用锁文件**（`USE_LOCK_FILE=0`），原因：
- `uv.lock` 语法与 `requirements.txt` 不同
- 锁文件对于跨平台部署可能有兼容性问题
- 直接使用 `requirements.txt` 更简单可靠

如果需要严格版本控制，建议在 `requirements.txt` 中明确指定版本号。

## 📦 部署后使用

### 激活环境

**Linux/macOS:**
```bash
source /opt/my_vibration_project/.venv/bin/activate
```

**Windows Git Bash:**
```bash
source /opt/my_vibration_project/.venv/Scripts/activate
```

### 运行项目

```bash
cd /opt/my_vibration_project
python main.py
```

### 管理依赖

```bash
# 查看已安装包
uv pip list

# 安装新包
uv pip install package_name

# 卸载包
uv pip uninstall package_name

# 更新包
uv pip install --upgrade package_name
```

### 退出环境

```bash
deactivate
```

## 🐛 故障排除

### 问题1：依赖安装失败

**原因**：某些包在 PyPI 上不可用或版本不匹配

**解决方案**：
1. 检查 `requirements_cleaned.txt` 中的失败包
2. 手动安装或更新版本：
   ```bash
   uv pip install package_name==version
   ```
3. 或使用 `requirements_core.txt`

### 问题2：无法激活虚拟环境

**原因**：路径分隔符在不同系统上不同

**解决方案**：
```bash
# 尝试两种路径
source /path/.venv/bin/activate    # Linux/macOS
source /path/.venv/Scripts/activate # Windows
```

### 问题3：uv 安装失败

**原因**：网络问题或权限不足

**解决方案**：
```bash
# 手动安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或使用 pip
pip install uv
```

### 问题4：rsync 命令不存在（Windows）

**解决方案**：脚本会自动回退到 `cp` 命令，无需处理

## 📝 文件说明

| 文件 | 说明 | 用途 |
|------|------|------|
| `deploy_env.sh` | 部署脚本 | 自动化环境部署 |
| `requirements.txt` | 完整依赖清单 | 导出的原始依赖 |
| `requirements_core.txt` | 核心依赖清单 | 精简的可用依赖 |
| `requirements_cleaned.txt` | 清理后依赖 | 脚本自动生成 |
| `README_deploy.md` | 本文档 | 部署说明 |

## 🚀 性能优势

使用 `uv` 相比传统 `pip` 的优势：
- ⚡ **速度快 10-100 倍**：依赖解析和安装极快
- 💾 **磁盘占用小**：智能缓存和硬链接
- 🔒 **更可靠**：更好的依赖解析算法
- 🌐 **跨平台**：完美支持 Windows/Linux/macOS

## 📚 参考资料

- [uv 官方文档](https://github.com/astral-sh/uv)
- [uv 快速入门](https://astral.sh/uv/guides/install-python.html)
- [Python 虚拟环境最佳实践](https://docs.python.org/3/tutorial/venv.html)

## 💡 最佳实践

1. **定期更新依赖**：使用 `uv pip list --outdated` 检查过期包
2. **使用版本锁定**：为生产环境明确指定版本号
3. **分离开发和生产依赖**：创建 `requirements-dev.txt`
4. **备份环境**：定期导出依赖清单
5. **使用虚拟环境**：永远不要在系统 Python 中安装包
