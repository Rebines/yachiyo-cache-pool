#!/bin/bash
# 🐙 一键启动 llama-server + 缓存代理（开包即用版）
# Yachiyo's Cache Pool - 克隆下来 bash start-cache.sh 就能跑

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OS_NAME="$(uname -s)"
IS_WINDOWS=false
case "$OS_NAME" in MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=true ;; esac
EXE=""
if $IS_WINDOWS; then EXE=".exe"; fi
LLAMA_SERVER_BIN="$SCRIPT_DIR/bin/llama-server$EXE"
MODEL_DIR="$SCRIPT_DIR/models"
DEFAULT_GGUF="$MODEL_DIR/embeddinggemma-300M-Q8_0.gguf"
GGUF_URL="https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf"
GGUF="$DEFAULT_GGUF"
PYTHON_DEPS=("flask" "requests")

# 选择可用的 Python（Windows 的 python3 可能是应用商店占位符）
PYTHON="python3"
if ! "$PYTHON" -c "print(1)" 2>/dev/null | grep -q '^1$'; then
    PYTHON="python"
fi
if ! "$PYTHON" -c "print(1)" 2>/dev/null | grep -q '^1$'; then
    echo "❌ 找不到可用的 Python（需要 python3 或 python）"
    exit 1
fi

# 从 config.yaml 读取配置（section 下的一级 key，值两端引号会被去掉）
# 不用 grep -P，兼容 Git Bash；找不到时输出空字符串
cfg_get() {
    awk -v section="$1" -v key="$2" '
        /^[^[:space:]#]/ { in_sec = (index($0, section ":") == 1) }
        in_sec && $1 == key ":" {
            sub(/^[^:]*:[[:space:]]*/, "")
            sub(/[[:space:]]+#.*$/, "")
            gsub(/^[[:space:]]+|[[:space:]]+$/, "")
            print
            exit
        }
    ' "$SCRIPT_DIR/config.yaml" 2>/dev/null | tr -d '\r' | tr -d '"'
}

PROXY_PORT=$(cfg_get proxy port | tr -cd '0-9')
if [ -z "$PROXY_PORT" ]; then PROXY_PORT=18791; fi
PROXY_HOST=$(cfg_get proxy host)
if [ -z "$PROXY_HOST" ]; then PROXY_HOST="127.0.0.1"; fi

# === 自定义 embedding 模型配置 ===
MODEL_PATH_CFG=$(cfg_get embedding model_path)
HF_REPO=$(cfg_get embedding hf_repo)
HF_FILE=$(cfg_get embedding hf_file)
_CFG_HF_TOKEN=$(cfg_get embedding hf_token)
HF_TOKEN="${_CFG_HF_TOKEN:-${HF_TOKEN:-}}"
_CFG_HF_ENDPOINT=$(cfg_get embedding hf_endpoint)
HF_ENDPOINT="${_CFG_HF_ENDPOINT:-${HF_ENDPOINT:-https://huggingface.co}}"
HF_ENDPOINT="${HF_ENDPOINT%/}"
POOLING=$(cfg_get embedding pooling)
if [ -z "$POOLING" ]; then POOLING="auto"; fi
EMBD_NORMALIZE=$(cfg_get embedding embd_normalize)
if [ -z "$EMBD_NORMALIZE" ]; then EMBD_NORMALIZE="2"; fi

echo "🐙 ヤチヨのキャッシュプロキシ～☆"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━"

# === 1. 检查/安装 Python 依赖 ===
echo "📦 检查 Python 依赖..."
MISSING_DEPS=false
for dep in "${PYTHON_DEPS[@]}"; do
    if ! "$PYTHON" -c "import $dep" 2>/dev/null; then
        MISSING_DEPS=true
        break
    fi
done

if $MISSING_DEPS; then
    echo "   → 安装 Python 依赖..."
    # Try with --break-system-packages first (Debian/Ubuntu), fallback to normal
    "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt" --break-system-packages -q 2>/dev/null || \
    "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt" -q 2>/dev/null || \
    "$PYTHON" -m pip install flask requests pyyaml --break-system-packages -q 2>/dev/null || \
    "$PYTHON" -m pip install flask requests pyyaml -q
fi
echo "   ✅ Python 依赖已就绪"

# === 检测后端配置 ===
BACKEND=$(cfg_get embedding backend)
if [ -z "$BACKEND" ]; then BACKEND="llama-server"; fi

if [ "$BACKEND" = "ollama" ]; then
    echo "🔧 后端: Ollama (跳过 llama-server 和模型下载)"
    echo "   → 确保 Ollama 正在运行: curl http://localhost:11434"
    echo "   → 确保 embedding 模型存在:"
    echo "     ollama pull embeddinggemma:300m-qat-q4_0"
    echo ""
else
    echo "🔧 后端: llama-server"
fi

# === 2. 检查/下载 llama-server（仅非 ollama 模式）===
if [ "$BACKEND" != "ollama" ]; then
echo "🔧 检查 llama-server..."

# === 检测系统环境（主要看有没有可用的 NVIDIA/CUDA）===
ARCH=$(uname -m)
ACCELERATOR=$(cfg_get embedding accelerator)
if [ -z "$ACCELERATOR" ]; then ACCELERATOR="auto"; fi
CUDA_BUILD_CFG=$(cfg_get embedding cuda_build)
if [ -z "$CUDA_BUILD_CFG" ]; then CUDA_BUILD_CFG="auto"; fi

# 版本比较：version_ge 13.0 12.4 → 真
version_ge() {
    awk -v a="$1" -v b="$2" 'BEGIN {
        na = split(a, x, "."); nb = split(b, y, ".");
        n = (na > nb) ? na : nb;
        for (i = 1; i <= n; i++) {
            if ((x[i] + 0) > (y[i] + 0)) exit 0;
            if ((x[i] + 0) < (y[i] + 0)) exit 1;
        }
        exit 0;
    }'
}

GPU_NAME=""; GPU_MEM=""; GPU_CC=""; DRIVER_CUDA=""
detect_nvidia() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi >/dev/null 2>&1 || return 1
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/[[:space:]]*$//')
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    # 多卡时取最低计算能力，保证选出的构建能覆盖所有卡
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | sort -n | head -1)
    DRIVER_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version:[[:space:]]*\([0-9][0-9.]*\).*/\1/p' | head -1)
    [ -n "$GPU_NAME" ]
}

USE_CUDA=false
CUDA_BACKEND_PRESENT=false
CUDA_PKG_VER=""
if [ "$OS_NAME" = "Darwin" ]; then
    echo "🖥️ macOS：llama.cpp 没有 CUDA 构建，使用 CPU 版"
elif [ "$ACCELERATOR" = "cpu" ]; then
    if detect_nvidia; then
        echo "🖥️ 检测到 NVIDIA GPU: $GPU_NAME（配置 accelerator=cpu）→ 使用 CPU 版"
    else
        echo "🖥️ 未检测到 NVIDIA GPU → 使用 CPU 版"
    fi
elif ! detect_nvidia; then
    if [ "$ACCELERATOR" = "cuda" ]; then
        echo "❌ 配置要求 CUDA，但未检测到可用的 NVIDIA GPU（nvidia-smi 不可用或执行失败）"
        exit 1
    fi
    echo "🖥️ 未检测到 NVIDIA GPU → 使用 CPU 版"
else
    echo "🖥️ 检测到 NVIDIA GPU: $GPU_NAME${GPU_MEM:+ (${GPU_MEM} MiB)}"
    echo "   驱动 CUDA: ${DRIVER_CUDA:-未知} / 计算能力: ${GPU_CC:-未知}"

    # 显式指定 cuda_build 时优先；auto 时按驱动版本和显卡能力选
    case "$CUDA_BUILD_CFG" in
        12|12.*) CUDA_PKG_VER=$(if $IS_WINDOWS; then echo "12.4"; else echo "12.8"; fi) ;;
        13|13.*) CUDA_PKG_VER="13.5" ;;
    esac
    if [ -n "$CUDA_PKG_VER" ]; then
        USE_CUDA=true
    elif [ -z "$DRIVER_CUDA" ]; then
        echo "   ⚠️ 读不到驱动 CUDA 版本，使用 CPU 版（可设 embedding.cuda_build: 12 强制）"
    else
        MIN_DRIVER_CUDA="12.4"
        if ! $IS_WINDOWS; then MIN_DRIVER_CUDA="12.8"; fi
        if [ -n "$GPU_CC" ] && version_ge "$GPU_CC" "10.0" && version_ge "$DRIVER_CUDA" "13.0"; then
            CUDA_PKG_VER="13.5"   # Blackwell 需要 CUDA 13 构建
        elif [ -n "$GPU_CC" ] && version_ge "$GPU_CC" "7.5" && version_ge "$DRIVER_CUDA" "13.5"; then
            CUDA_PKG_VER="13.5"
        elif version_ge "$DRIVER_CUDA" "$MIN_DRIVER_CUDA"; then
            CUDA_PKG_VER=$(if $IS_WINDOWS; then echo "12.4"; else echo "12.8"; fi)
        fi
        if [ -n "$CUDA_PKG_VER" ]; then
            USE_CUDA=true
        else
            echo "   ⚠️ 驱动 CUDA $DRIVER_CUDA 低于所需版本 $MIN_DRIVER_CUDA，使用 CPU 版"
        fi
    fi

    if $USE_CUDA; then
        # arm64 目前只发布了 CUDA 13.5 包
        case "$ARCH" in aarch64|arm64) CUDA_PKG_VER="13.5" ;; esac
        if [ -n "$GPU_CC" ] && [ "$CUDA_PKG_VER" = "13.5" ] && ! version_ge "$GPU_CC" "7.5"; then
            echo "   ⚠️ 计算能力 $GPU_CC 不支持 CUDA 13 构建；可设置 embedding.cuda_build: 12"
        fi
        echo "   → 使用 CUDA $CUDA_PKG_VER 版 llama.cpp"
    fi
fi

if [ ! -f "$LLAMA_SERVER_BIN" ]; then
    echo "   → 未找到本地 llama-server，尝试自动下载..."
    mkdir -p "$SCRIPT_DIR/bin"

    # 动态获取最新版本号（新版 llama.cpp 用 nightly-tag.txt 指向具体 build）
    echo "   → 查询最新版本..."
    LLAMA_TAG=$(curl -sL "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" 2>/dev/null | "$PYTHON" -c "import json,sys; d=json.load(sys.stdin); print(d.get('tag_name',''))" 2>/dev/null)
    if [ "$LLAMA_TAG" = "v1.0.0" ] || [ -z "$LLAMA_TAG" ]; then
        NIGHTLY_TAG=$(curl -sL "https://github.com/ggml-org/llama.cpp/releases/download/v1.0.0/nightly-tag.txt" 2>/dev/null | tr -d '\r\n')
        if [ -n "$NIGHTLY_TAG" ]; then LLAMA_TAG="$NIGHTLY_TAG"; fi
    fi
    if [ -z "$LLAMA_TAG" ]; then
        echo "   ⚠️ 无法获取最新版本号，使用默认 b12359"
        LLAMA_TAG="b12359"
    fi
    echo "   → 版本: $LLAMA_TAG"

    PKG=""
    PKG_CUDART=""
    if $IS_WINDOWS; then
        case "$ARCH" in
            aarch64|arm64)
                if $USE_CUDA; then
                    PKG="llama-${LLAMA_TAG}-bin-win-cuda-${CUDA_PKG_VER}-arm64.zip"
                    PKG_CUDART="cudart-llama-bin-win-cuda-${CUDA_PKG_VER}-arm64.zip"
                else
                    PKG="llama-${LLAMA_TAG}-bin-win-cpu-arm64.zip"
                fi
                ;;
            x86_64|amd64)
                if $USE_CUDA; then
                    PKG="llama-${LLAMA_TAG}-bin-win-cuda-${CUDA_PKG_VER}-x64.zip"
                    PKG_CUDART="cudart-llama-bin-win-cuda-${CUDA_PKG_VER}-x64.zip"
                else
                    PKG="llama-${LLAMA_TAG}-bin-win-cpu-x64.zip"
                fi
                ;;
            *)
                echo "   ⚠️ Windows 架构 $ARCH 没有预编译的 llama-server"
                echo "   可从 GitHub Releases 手动下载 win-cpu 包，解压到 $SCRIPT_DIR/bin/"
                echo "   或改用 Ollama（编辑 config.yaml 设置 backend: ollama）"
                exit 1
                ;;
        esac
    else
        case "$ARCH" in
            aarch64|arm64)
                if [ "$OS_NAME" = "Darwin" ]; then
                    PKG="llama-${LLAMA_TAG}-bin-macos-arm64.tar.gz"
                elif $USE_CUDA; then
                    PKG="llama-${LLAMA_TAG}-bin-ubuntu-cuda-${CUDA_PKG_VER}-arm64.tar.gz"
                    PKG_CUDART="cudart-llama-${LLAMA_TAG}-bin-ubuntu-cuda-${CUDA_PKG_VER}-arm64.tar.gz"
                else
                    PKG="llama-${LLAMA_TAG}-bin-ubuntu-arm64.tar.gz"
                fi
                ;;
            x86_64|amd64)
                if [ "$OS_NAME" = "Darwin" ]; then
                    PKG="llama-${LLAMA_TAG}-bin-macos-x64.tar.gz"
                elif $USE_CUDA; then
                    PKG="llama-${LLAMA_TAG}-bin-ubuntu-cuda-${CUDA_PKG_VER}-x64.tar.gz"
                    PKG_CUDART="cudart-llama-${LLAMA_TAG}-bin-ubuntu-cuda-${CUDA_PKG_VER}-x64.tar.gz"
                else
                    PKG="llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz"
                fi
                ;;
            *)
                USE_CUDA=false
                echo "   ⚠️ 架构 $ARCH 没有预编译的 llama-server"
                echo "   尝试从系统寻找..."
                for loc in /usr/local/lib/ollama/llama-server /usr/bin/llama-server /usr/local/bin/llama-server; do
                    if [ -f "$loc" ]; then
                        echo "   → 使用 $loc"
                        cp "$loc" "$LLAMA_SERVER_BIN"
                        chmod +x "$LLAMA_SERVER_BIN"
                        break
                    fi
                done
                if [ ! -f "$LLAMA_SERVER_BIN" ]; then
                    echo "❌ 找不到 llama-server。可以:"
                    echo "   1. 从 GitHub Releases 手动下载: https://github.com/ggml-org/llama.cpp/releases"
                    echo "   2. 或改用 Ollama（编辑 config.yaml 设置 backend: ollama）"
                    exit 1
                fi
                ;;
        esac
    fi
    
    # 下载并解压 release 包到 bin/（tar 包有顶层目录；Windows zip 是平铺的）
    fetch_pkg() {
        local pkg="$1" tmp="/tmp/$1"
        local url="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/$pkg"
        echo "   → 下载 $pkg ..."
        echo "     从 GitHub Releases 下载，国内可能稍慢，请稍候..."
        echo "     (若长时间无响应可按 Ctrl+C 取消)"
        # 大文件在部分网络下会被中途掐断，开启重试并支持断点续传
        rm -f "$tmp"
        if command -v curl &>/dev/null; then
            if ! curl -L -f -C - --retry 10 --retry-all-errors --retry-delay 1 \
                --progress-bar -o "$tmp" "$url"; then
                rm -f "$tmp"
                return 1
            fi
        else
            if ! wget --no-check-certificate --show-progress -O "$tmp" "$url"; then
                rm -f "$tmp"
                return 1
            fi
        fi
        mkdir -p "$SCRIPT_DIR/bin"
        case "$tmp" in
            *.zip)
                if command -v unzip &>/dev/null; then
                    unzip -o -q "$tmp" -d "$SCRIPT_DIR/bin"
                else
                    powershell -NoProfile -Command "Expand-Archive -LiteralPath '$(cygpath -w "$tmp")' -DestinationPath '$(cygpath -w "$SCRIPT_DIR/bin")' -Force"
                fi
                ;;
            *.tar.gz)
                tar xzf "$tmp" -C "$SCRIPT_DIR/bin" --strip-components 1 2>/dev/null
                ;;
        esac
        rm -f "$tmp"
        return 0
    }

    if [ -n "$PKG" ]; then
        if fetch_pkg "$PKG"; then
            if [ -f "$LLAMA_SERVER_BIN" ]; then
                chmod +x "$LLAMA_SERVER_BIN"
                echo "   ✅ llama-server 下载成功 ($(du -h "$LLAMA_SERVER_BIN" | cut -f1))"
            else
                # 再试试用 find 找一下
                FOUND=$(find "$SCRIPT_DIR/bin" -name "llama-server$EXE" -type f 2>/dev/null | head -1)
                if [ -n "$FOUND" ]; then
                    mv "$FOUND" "$LLAMA_SERVER_BIN"
                    chmod +x "$LLAMA_SERVER_BIN"
                    echo "   ✅ llama-server 解压成功"
                else
                    echo "   ⚠️ 下载包可能不完整，删除重试..."
                    rm -rf "$SCRIPT_DIR/bin/"*
                    echo "   ❌ 下载失败。可从 GitHub 手动下载:"
                    echo "      https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG/$PKG"
                    echo "     然后解压出 llama-server 放到 $SCRIPT_DIR/bin/"
                    echo "   🔄 或改用 Ollama 后端（编辑 config.yaml 设置 backend: ollama）"
                    exit 1
                fi
            fi
            # CUDA 主包不含 CUDA 运行库，需要配套下载（cublas/cudart）
            if [ -n "$PKG_CUDART" ] && [ -f "$LLAMA_SERVER_BIN" ]; then
                if fetch_pkg "$PKG_CUDART"; then
                    CUDA_BACKEND_PRESENT=true
                    echo "   ✅ CUDA 运行库已就绪"
                else
                    echo "   ⚠️ CUDA 运行库下载失败: $PKG_CUDART"
                    echo "   可手动从 https://github.com/ggml-org/llama.cpp/releases/tag/$LLAMA_TAG 下载"
                    echo "   并解压到 $SCRIPT_DIR/bin/，或设置 embedding.accelerator: cpu 回退 CPU 模式"
                    exit 1
                fi
            fi
        else
            echo "   ⚠️ 下载失败，尝试从系统寻找..."
            for loc in /usr/local/lib/ollama/llama-server /usr/bin/llama-server /usr/local/bin/llama-server; do
                if [ -f "$loc" ]; then
                    cp "$loc" "$LLAMA_SERVER_BIN"
                    chmod +x "$LLAMA_SERVER_BIN"
                    echo "   → 使用 $loc"
                    break
                fi
            done
            if [ ! -f "$LLAMA_SERVER_BIN" ]; then
                echo "❌ 无法获取 llama-server。可以:"
                echo "   1. 从 GitHub Releases 手动下载: https://github.com/ggml-org/llama.cpp/releases"
                echo "   2. 或改用 Ollama（编辑 config.yaml 设置 backend: ollama）"
                exit 1
            fi
            if $USE_CUDA; then
                echo "   ⚠️ 回退到系统 llama-server，可能没有 CUDA 加速"
            fi
        fi
    fi
else
    echo "   ✅ 本地 llama-server 已就绪"
    if $USE_CUDA; then
        if [ -f "$SCRIPT_DIR/bin/ggml-cuda.dll" ] || ls "$SCRIPT_DIR/bin"/libggml-cuda.so* >/dev/null 2>&1; then
            CUDA_BACKEND_PRESENT=true
            echo "   → 已包含 CUDA 后端"
        else
            echo "   ⚠️ 现有 llama-server 未包含 CUDA 后端；删除 $SCRIPT_DIR/bin/ 后重跑可自动下载 CUDA 版"
        fi
    fi
fi
fi

# === 3. 检查/下载 GGUF 模型 ===
# 把配置里的路径展开成 Git Bash 可用的本地路径：
#   ~/xxx           → $HOME/xxx
#   D:\models\x.gguf → /d/models/x.gguf（Windows 下用 cygpath 转换）
#   相对路径         → 相对本仓库根目录
resolve_model_path() {
    local p="$1"
    p="${p/#\~/$HOME}"
    if $IS_WINDOWS && [[ "$p" =~ ^[A-Za-z]: ]]; then
        p="$(cygpath -u "$p")"
    fi
    case "$p" in
        /*) ;;
        *) p="$SCRIPT_DIR/$p" ;;
    esac
    printf '%s' "$p"
}

# 校验是不是真正的 GGUF（避免把 404 页面当成模型）
is_gguf_file() {
    [ -f "$1" ] && [ "$(head -c 4 "$1" 2>/dev/null)" = "GGUF" ]
}

# 查询 HF 仓库的 GGUF 文件列表，自动挑一个（需要 token 时带上）
select_hf_file() {
    "$PYTHON" - "$HF_REPO" "$HF_ENDPOINT" "$HF_TOKEN" <<'PY'
import json, sys, urllib.request

repo, endpoint, token = sys.argv[1], sys.argv[2], sys.argv[3]
req = urllib.request.Request(endpoint.rstrip("/") + "/api/models/" + repo)
if token:
    req.add_header("Authorization", "Bearer " + token)
with urllib.request.urlopen(req, timeout=30) as resp:
    data = json.load(resp)
files = [s.get("rfilename", "") for s in data.get("siblings", [])]
files = [f for f in files if f.lower().endswith(".gguf") and "mmproj" not in f.lower()]

def rank(f):
    n = f.lower()
    for i, tag in enumerate(("q8_0", "q6_k", "q5_k_m", "q4_k_m")):
        if tag in n:
            return i
    return 9

files.sort(key=lambda f: (rank(f), len(f), f))
print(files[0] if files else "")
PY
}

# 下载 GGUF：先落 .part，校验魔数后再改名，避免半截文件被当成可用模型
download_gguf() {
    local url="$1" dest="$2" tmp="$2.part"
    mkdir -p "$(dirname "$dest")"
    if command -v curl &>/dev/null; then
        if [ -n "$HF_TOKEN" ]; then
            curl -L -f -C - --retry 10 --retry-all-errors --retry-delay 1 \
                --progress-bar -H "Authorization: Bearer $HF_TOKEN" -o "$tmp" "$url"
        else
            curl -L -f -C - --retry 10 --retry-all-errors --retry-delay 1 \
                --progress-bar -o "$tmp" "$url"
        fi
    elif [ -n "$HF_TOKEN" ]; then
        wget --header="Authorization: Bearer $HF_TOKEN" --show-progress -O "$tmp" "$url"
    else
        wget --show-progress -O "$tmp" "$url"
    fi
    if is_gguf_file "$tmp"; then
        mv -f "$tmp" "$dest"
        return 0
    fi
    rm -f "$tmp"
    return 1
}

if [ "$BACKEND" != "ollama" ]; then
echo "🧠 检查 embedding 模型..."
if [ -n "$MODEL_PATH_CFG" ]; then
    # --- 自定义本地 GGUF ---
    GGUF="$(resolve_model_path "$MODEL_PATH_CFG")"
    echo "   → 使用自定义本地模型: $GGUF"
    if ! is_gguf_file "$GGUF"; then
        echo "❌ 找不到有效的 GGUF 文件: $GGUF"
        echo "   请检查 config.yaml 的 embedding.model_path"
        echo "   （支持绝对路径、~/ 开头、相对仓库根目录的路径）"
        exit 1
    fi
    echo "   ✅ 模型已就绪 ($(du -h "$GGUF" | cut -f1))"
elif [ -n "$HF_REPO" ]; then
    # --- 自定义 HuggingFace 仓库 ---
    echo "   → 使用 HuggingFace 仓库: $HF_REPO"
    if [ -z "$HF_FILE" ]; then
        echo "   → 未指定 hf_file，查询仓库文件列表..."
        HF_FILE="$(select_hf_file 2>/dev/null || true)"
        if [ -z "$HF_FILE" ]; then
            echo "❌ 无法从 $HF_REPO 获取 GGUF 文件列表"
            echo "   请检查仓库 ID / 网络，或在 config.yaml 显式设置 embedding.hf_file"
            exit 1
        fi
    fi
    GGUF="$MODEL_DIR/$(basename "$HF_FILE")"
    echo "   → 仓库文件: $HF_FILE"
    if is_gguf_file "$GGUF"; then
        echo "   ✅ 模型已就绪 ($(du -h "$GGUF" | cut -f1))"
    else
        echo "   ⬇️  下载 $HF_FILE ..."
        echo "     从 $HF_ENDPOINT 下载，国内可能较慢，耐心等待..."
        HF_URL_PATH=$("$PYTHON" -c "import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe='/'))" "$HF_REPO/resolve/main/$HF_FILE")
        HF_URL="$HF_ENDPOINT/$HF_URL_PATH"
        if download_gguf "$HF_URL" "$GGUF"; then
            echo "   ✅ 模型下载成功 ($(du -h "$GGUF" | cut -f1))"
        else
            echo "❌ 下载失败。请手动下载:"
            echo "   $HF_URL"
            echo "   放到 $GGUF"
            exit 1
        fi
    fi
else
    # --- 默认 embeddinggemma ---
    GGUF="$DEFAULT_GGUF"
    if ! is_gguf_file "$GGUF"; then
        echo "   → 未找到可用的本地模型，自动下载 (~314MB)..."
        echo "   ⬇️  下载 embeddinggemma-300M-Q8_0 (~314MB) ..."
        echo "     从 HuggingFace 下载，国内可能较慢，耐心等待..."
        echo "     (若长时间无进度可按 Ctrl+C 取消)"
        if download_gguf "$GGUF_URL" "$GGUF"; then
            echo "   ✅ 模型下载成功 ($(du -h "$GGUF" | cut -f1))"
        else
            echo "❌ 下载失败。请手动下载:"
            echo "   $GGUF_URL"
            echo "   放到 $GGUF"
            exit 1
        fi
    else
        echo "   ✅ 模型已就绪 ($(du -h "$GGUF" | cut -f1))"
    fi
fi
fi

# === 4. 启动 llama-server（仅非 ollama 模式）===
if [ "$BACKEND" != "ollama" ]; then
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    # 已有服务在跑：确认它加载的就是当前配置的模型，否则新配置不会生效
    RUNNING_MODEL=$(curl -s --max-time 3 http://localhost:8080/props \
        | "$PYTHON" -c "import json,sys; print(json.load(sys.stdin).get('model_path',''))" 2>/dev/null || true)
    if [ -n "$RUNNING_MODEL" ]; then
        RUNNING_BASE="${RUNNING_MODEL//\\//}"
        RUNNING_BASE="${RUNNING_BASE##*/}"
        CONFIG_BASE=$(basename "$GGUF")
        if [ "$RUNNING_BASE" != "$CONFIG_BASE" ]; then
            echo "⚠️ 端口 8080 上已有 llama-server 在运行，但加载的是另一个模型:"
            echo "   当前: $RUNNING_MODEL"
            echo "   配置: $GGUF"
            echo "   新配置不会自动生效，请先停止旧服务再重新运行本脚本:"
            echo "   Windows: taskkill /F /IM llama-server.exe"
            echo "   Linux/macOS: pkill -f llama-server"
            exit 1
        fi
    fi
    echo "✅ llama.cpp server (端口 8080) 已在运行中"
else
    echo "🚀 启动 llama.cpp embedding server..."
    # 新版 llama.cpp 已移除 --no-mmap，旧版仍支持：探测后再传
    LLAMA_ARGS=()
    if "$LLAMA_SERVER_BIN" --help 2>&1 | grep -q -- "--no-mmap"; then
        LLAMA_ARGS+=(--no-mmap)
    fi
    # auto = 不传 --pooling，让 llama-server 用模型自带默认（不同模型池化方式不同）
    case "$POOLING" in
        ""|auto|default|model) ;;
        *) LLAMA_ARGS+=(--pooling "$POOLING") ;;
    esac
    LLAMA_ARGS+=(--embd-normalize "$EMBD_NORMALIZE")
    # Windows 原生程序需要 Windows 风格路径
    GGUF_ARG="$GGUF"
    if $IS_WINDOWS; then GGUF_ARG="$(cygpath -w "$GGUF")"; fi
    nohup "$LLAMA_SERVER_BIN" \
        --model "$GGUF_ARG" --port 8080 --host 127.0.0.1 \
        --embedding --ctx-size 512 --batch-size 512 \
        --threads 4 --no-webui "${LLAMA_ARGS[@]}" \
        --log-disable \
        > /dev/null 2>&1 &
    LLAMA_PID=$!

    READY=false
    for _ in $(seq 1 20); do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            READY=true
            break
        fi
        sleep 1
    done
    if $READY; then
        echo "   ✅ PID: $LLAMA_PID"
    else
        echo "❌ llama-server 启动失败"
        exit 1
    fi
fi
fi

# === 5. 启动缓存代理 ===
if curl -s "http://localhost:$PROXY_PORT/health" > /dev/null 2>&1; then
    echo "✅ 缓存代理 (端口 $PROXY_PORT) 已在运行中"
else
    echo "🚀 启动缓存代理..."
    nohup "$PYTHON" "$SCRIPT_DIR/cache_proxy.py" > /tmp/cache_proxy.log 2>&1 &
    PROXY_READY=false
    for _ in $(seq 1 15); do
        if curl -s "http://localhost:$PROXY_PORT/health" > /dev/null 2>&1; then
            PROXY_READY=true
            break
        fi
        sleep 1
    done
    if $PROXY_READY; then
        echo "   ✅ 缓存代理已就绪 ($PROXY_HOST:$PROXY_PORT)"
    else
        echo "❌ 代理启动失败，看日志: tail /tmp/cache_proxy.log"
        exit 1
    fi
fi

# === 6. 完成 ===
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 全部就绪！"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   🧠 Embedding:  http://localhost:8080/v1/embeddings"
if [ "$BACKEND" != "ollama" ]; then
    echo "      模型:       $GGUF"
    echo "      池化:       $POOLING (normalize=$EMBD_NORMALIZE)"
    if $USE_CUDA && $CUDA_BACKEND_PRESENT; then
        echo "      加速:       CUDA $CUDA_PKG_VER ($GPU_NAME)"
    elif $USE_CUDA; then
        echo "      加速:       CPU（检测到 CUDA，但 bin/ 里是 CPU 版）"
    else
        echo "      加速:       CPU"
    fi
fi
echo "   🚀 缓存代理:   http://localhost:$PROXY_PORT"
echo "   💾 数据库:     $SCRIPT_DIR/cache.db"
echo "   📊 命中日志:   $SCRIPT_DIR/cache_hit.log"
echo ""
echo "📋 使用说明:"
echo "   将任何 OpenAI 兼容客户端的 baseUrl 改为:"
echo "   http://localhost:$PROXY_PORT"
echo "   API Key 会自动透传"
echo ""
echo "🔍 查看日志:   tail -f /tmp/cache_proxy.log"
echo "📊 缓存统计:   curl -s http://localhost:$PROXY_PORT/health | $PYTHON -m json.tool"
echo "🔬 测试搜索:   cd $SCRIPT_DIR && bash cache.sh search '试试'"
echo ""
echo "🐙 感謝 感激 雨アラモード～☆"
