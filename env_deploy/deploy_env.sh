#!/bin/bash
# 功能：一键用uv部署Python环境（支持「从零部署」/「克隆本地环境」两种模式）
# 适用：Linux/macOS（Windows需用WSL/Git Bash）
# 使用前：修改下方【用户配置区】的路径和参数

# ======================== 【用户配置区】（仅需修改这里） ========================
# 1. 目标部署目录（脚本会自动创建，建议绝对路径）
PROJECT_DIR="/opt/my_vibration_project"

# 2. 本地项目源代码路径（需要复制到目标目录的项目代码）
#    如果为空，则只部署环境，不复制代码
SOURCE_PROJECT_DIR=""  # 示例："/home/user/my_local_project"

# 3. requirements.txt的路径（本地已导出的依赖清单）
#    默认使用脚本所在目录的 requirements.txt
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

# 4. 是否生成/使用uv.lock锁文件（1=是，0=否，推荐0，因为uv.lock用法较复杂）
USE_LOCK_FILE=0

# 5. Python版本指定（留空则使用系统默认，示例：3.11、3.10）
PYTHON_VERSION=""

# ========== 本地环境克隆相关配置 ==========
# 6. 是否从本地环境克隆（1=克隆本地环境，0=从零部署，二选一）
USE_LOCAL_ENV=0
# 7. 本地已有Python环境的路径（仅当USE_LOCAL_ENV=1时生效）
#    示例：Anaconda环境 /home/user/anaconda3/envs/my_env
#    示例：venv环境 /home/user/old_project/.venv
LOCAL_ENV_PATH=""
# ===============================================================================

# ======================== 【脚本核心逻辑】（无需修改） ========================
# 错误处理：遇到错误立即退出，避免无效执行
set -euo pipefail

# 步骤1：创建部署目录并检查/清理requirements.txt
echo -e "\n【步骤1/7】创建部署目录并处理依赖清单..."
mkdir -p "$PROJECT_DIR"

# 检查 requirements.txt 是否存在
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "❌ 错误：未找到requirements.txt！请确认路径：$REQUIREMENTS_FILE"
    echo -e "   提示：请先将本地导出的requirements.txt放到脚本所在目录下，再重新运行脚本。"
    exit 1
fi

# 清理 requirements.txt：移除本地文件路径依赖（@ file:///...）
echo -e "📌 清理 requirements.txt 中的本地文件路径依赖..."
CLEANED_REQUIREMENTS="$PROJECT_DIR/requirements_cleaned.txt"
grep -v "@ file:///" "$REQUIREMENTS_FILE" > "$CLEANED_REQUIREMENTS" || true

# 统计清理结果
ORIGINAL_COUNT=$(wc -l < "$REQUIREMENTS_FILE")
CLEANED_COUNT=$(wc -l < "$CLEANED_REQUIREMENTS")
REMOVED_COUNT=$((ORIGINAL_COUNT - CLEANED_COUNT))

echo -e "✅ 依赖清单处理完成："
echo -e "   原始依赖: $ORIGINAL_COUNT 个"
echo -e "   可用依赖: $CLEANED_COUNT 个"
echo -e "   已移除本地路径依赖: $REMOVED_COUNT 个"
echo -e "   清理后文件: $CLEANED_REQUIREMENTS"

# 更新 REQUIREMENTS_FILE 指向清理后的文件
REQUIREMENTS_FILE="$CLEANED_REQUIREMENTS"

# 步骤2：安装uv（如果未安装）
echo -e "\n【步骤2/7】检查并安装uv包管理器..."
if ! command -v uv &> /dev/null; then
    echo -e "⚠️  未检测到uv，开始自动安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 临时加载uv到当前终端（避免重启终端）
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo -e "✅ uv已安装：$(uv --version)"
fi

# ========== 步骤3：检查并验证本地环境 ==========
if [ $USE_LOCAL_ENV -eq 1 ]; then
    echo -e "\n【步骤3/7】检查本地克隆环境的有效性..."
    # 检查本地环境路径是否存在
    if [ ! -d "$LOCAL_ENV_PATH" ]; then
        echo -e "❌ 错误：指定的本地环境路径不存在！路径：$LOCAL_ENV_PATH"
        exit 1
    fi
    # 检查是否是有效的Python环境（验证是否有python解释器）
    if [ -f "$LOCAL_ENV_PATH/bin/python" ]; then
        LOCAL_PYTHON="$LOCAL_ENV_PATH/bin/python"
    elif [ -f "$LOCAL_ENV_PATH/Scripts/python.exe" ]; then
        LOCAL_PYTHON="$LOCAL_ENV_PATH/Scripts/python.exe"
    else
        echo -e "❌ 错误：指定的路径不是有效的Python环境（未找到python解释器）！"
        exit 1
    fi
    # 验证本地环境的Python版本
    LOCAL_PY_VERSION=$($LOCAL_PYTHON --version 2>&1)
    echo -e "✅ 本地环境验证成功：$LOCAL_ENV_PATH"
    echo -e "   📌 本地环境Python版本：$LOCAL_PY_VERSION"
fi

# 步骤4：创建/克隆uv虚拟环境
echo -e "\n【步骤4/7】创建/克隆uv虚拟环境..."
cd "$PROJECT_DIR"
if [ $USE_LOCAL_ENV -eq 1 ]; then
    # 从本地环境克隆虚拟环境（复用已有依赖）
    echo -e "📌 从本地环境克隆：$LOCAL_ENV_PATH"
    uv venv --clone "$LOCAL_ENV_PATH"
else
    # 从零创建空的虚拟环境
    if [ -n "$PYTHON_VERSION" ]; then
        echo -e "📌 指定Python版本：$PYTHON_VERSION"
        uv venv --python "$PYTHON_VERSION"
    else
        uv venv
    fi
fi
echo -e "✅ 虚拟环境创建/克隆成功：$PROJECT_DIR/.venv"

# 步骤5：安装/更新依赖
echo -e "\n【步骤5/7】安装/更新依赖（uv速度远快于pip）..."

# 激活虚拟环境
source "$PROJECT_DIR/.venv/bin/activate" || source "$PROJECT_DIR/.venv/Scripts/activate" 2>/dev/null || {
    echo -e "⚠️  警告：无法激活虚拟环境，尝试直接使用 uv pip..."
}

if [ $USE_LOCK_FILE -eq 1 ]; then
    # 生成锁文件（固定依赖版本，保证环境一致）
    echo -e "📌 生成 uv.lock 锁文件..."
    uv pip compile "$REQUIREMENTS_FILE" -o "$PROJECT_DIR/uv.lock"
    # 使用锁文件同步依赖（确保版本完全一致）
    echo -e "📌 使用锁文件同步依赖..."
    uv pip sync "$PROJECT_DIR/uv.lock"
else
    # 直接安装 requirements.txt（推荐方式）
    echo -e "📌 安装依赖包..."
    uv pip install -r "$REQUIREMENTS_FILE"
fi

echo -e "✅ 依赖安装/更新完成！"

# 步骤6：复制项目代码（可选）
if [ -n "$SOURCE_PROJECT_DIR" ] && [ -d "$SOURCE_PROJECT_DIR" ]; then
    echo -e "\n【步骤6/7】复制项目代码到目标目录..."
    echo -e "📌 源目录：$SOURCE_PROJECT_DIR"
    echo -e "📌 目标目录：$PROJECT_DIR"
    
    # 复制项目文件（排除虚拟环境、缓存等）
    rsync -av --progress \
        --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='.idea' \
        --exclude='*.egg-info' \
        --exclude='results' \
        --exclude='data' \
        "$SOURCE_PROJECT_DIR/" "$PROJECT_DIR/" 2>/dev/null || {
        echo -e "⚠️  rsync 不可用，使用 cp 命令复制..."
        cp -r "$SOURCE_PROJECT_DIR"/* "$PROJECT_DIR/" 2>/dev/null || true
    }
    
    echo -e "✅ 项目代码复制完成！"
else
    echo -e "\n【步骤6/7】跳过项目代码复制（未配置源目录）"
fi

# 步骤7：验证环境
echo -e "\n【步骤7/7】验证环境是否正常..."
# 激活虚拟环境并检查依赖
source "$PROJECT_DIR/.venv/bin/activate" || source "$PROJECT_DIR/.venv/Scripts/activate" 2>/dev/null

echo -e "📌 当前虚拟环境Python版本：$(python --version)"
echo -e "📌 已安装的核心依赖列表（前20个）："
uv pip list | head -20

# 尝试退出虚拟环境
deactivate 2>/dev/null || true

# 最终提示
echo -e "\n" + "="*80
echo -e "🎉 环境部署全部完成！"
echo -e "="*80
echo -e "\n🔧 后续使用方式："
echo -e "   1. 激活环境："
echo -e "      Linux/macOS: source $PROJECT_DIR/.venv/bin/activate"
echo -e "      Windows Git Bash: source $PROJECT_DIR/.venv/Scripts/activate"
echo -e "   2. 运行项目：cd $PROJECT_DIR && python 你的脚本名.py"
echo -e "   3. 查看依赖：uv pip list（激活环境后）"
echo -e "   4. 退出环境：deactivate"

if [ $USE_LOCAL_ENV -eq 1 ]; then
    echo -e "\n📌 克隆说明：本次环境基于本地 $LOCAL_ENV_PATH 克隆，已复用原有依赖。"
fi

if [ -n "$SOURCE_PROJECT_DIR" ] && [ -d "$SOURCE_PROJECT_DIR" ]; then
    echo -e "\n📌 项目代码已从 $SOURCE_PROJECT_DIR 复制到 $PROJECT_DIR"
fi

echo -e "\n⚠️  注意事项："
echo -e "   - 清理后的依赖文件：$CLEANED_REQUIREMENTS"
echo -e "   - 已移除 $REMOVED_COUNT 个无法在新环境使用的本地路径依赖"
echo -e "   - 如有依赖缺失，请手动安装：uv pip install 包名"
echo -e "\n" + "="*80