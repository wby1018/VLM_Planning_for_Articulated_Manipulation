#!/usr/bin/env bash
# run_all.sh — 一键启动 VLM-based 操作流水线
#
# 用法:
#   ./run_all.sh            # 默认柜子 40147
#   ./run_all.sh 44817      # 切换到 44817
#   ./run_all.sh 46230      # 切换到 46230
#
# 流程:
#   1) 后台启 det_pipeline.py，等 :8001 就绪
#   2) 后台启 action_server.py，等 ZMQ :5555 就绪
#   3) 前台启 client_sapien_<id>.py（你看到 SAPIEN 窗口后聚焦窗口按空格触发）
#   4) Ctrl+C 或者 SAPIEN 窗口关闭后，自动清理两个后台进程

set -u
CABINET_ID="${1:-40147}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="/home/harvardair/Downloads/yes/etc/profile.d/conda.sh"
ENV_NAME="owlsam"
DET_PORT=8001
ZMQ_PORT=5555
RECON_PORT=8002
RECON_PATH="${RECONSTRUCTION_PIPELINE_PATH:-/home/harvardair/.yifeng/reconstruction_activeperception}"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"
DET_LOG="${LOG_DIR}/det_pipeline.log"
SRV_LOG="${LOG_DIR}/action_server.log"
RECON_LOG="${LOG_DIR}/reconstruction.log"

# 校验柜子 ID
case "${CABINET_ID}" in
  40147|44817|46230) ;;
  *) echo "❌ 不认识的柜子 ID: ${CABINET_ID}（支持 40147 / 44817 / 46230）"; exit 1 ;;
esac
CLIENT_SCRIPT="${REPO_DIR}/client_sapien_${CABINET_ID}.py"
[[ -f "${CLIENT_SCRIPT}" ]] || { echo "❌ 找不到 ${CLIENT_SCRIPT}"; exit 1; }

# 激活 conda 环境
[[ -f "${CONDA_SH}" ]] || { echo "❌ 找不到 conda: ${CONDA_SH}"; exit 1; }
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${ENV_NAME}" || { echo "❌ 无法激活 conda env '${ENV_NAME}'"; exit 1; }
echo "✅ 已激活 conda env: ${ENV_NAME}  (python: $(python -V 2>&1))"

# Source the OpenAI key file if present (for action_server VLM call_vlm + the
# sidecar's Phase D update_joint_vlm). nohup'd / non-interactive shells skip
# ~/.bashrc's source line, so we do it explicitly here.
if [[ -f "${HOME}/.openai_env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.openai_env"
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    echo "✅ OPENAI_API_KEY loaded (length=${#OPENAI_API_KEY}, prefix=${OPENAI_API_KEY:0:7}…)"
  fi
else
  echo "⚠️  ~/.openai_env not found — VLM (action_server.call_vlm) and Phase D"
  echo "   (sidecar.update_joint_vlm) will both fall back to defaults."
fi

cd "${REPO_DIR}"

# ── 端口占用预检查 ──────────────────────────────────────────────
if ss -ltn | awk '{print $4}' | grep -qE ":${DET_PORT}$"; then
  echo "⚠️  端口 :${DET_PORT} 已被占用：$(ss -ltnp 2>/dev/null | awk -v p=":${DET_PORT}$" '$4 ~ p {print $0}')"
  echo "   det_pipeline.py 里写的是 ${DET_PORT}（之前从 8000 改过来的）。"
  echo "   请先释放该端口（kill 占用进程）或修改脚本里的 DET_PORT。"
  exit 1
fi
if ss -ltn | awk '{print $4}' | grep -qE ":${ZMQ_PORT}$"; then
  echo "⚠️  端口 :${ZMQ_PORT} 已被占用，action_server 会 bind 失败。"
  exit 1
fi
if ss -ltn | awk '{print $4}' | grep -qE ":${RECON_PORT}$"; then
  echo "⚠️  端口 :${RECON_PORT} 已被占用，reconstruction_server 会 bind 失败。"
  exit 1
fi
[[ -d "${RECON_PATH}" ]] || { echo "❌ RECONSTRUCTION_PIPELINE_PATH 不存在: ${RECON_PATH}"; exit 1; }

# ── 清理函数 (Ctrl+C 或者退出时) ────────────────────────────────
DET_PID=""
SRV_PID=""
RECON_PID=""
cleanup() {
  echo
  echo "🧹 清理后台进程..."
  if [[ -n "${DET_PID}" ]] && kill -0 "${DET_PID}" 2>/dev/null; then
    echo "   stopping det_pipeline (PID ${DET_PID})"
    kill "${DET_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SRV_PID}" ]] && kill -0 "${SRV_PID}" 2>/dev/null; then
    echo "   stopping action_server (PID ${SRV_PID})"
    kill "${SRV_PID}" 2>/dev/null || true
  fi
  if [[ -n "${RECON_PID}" ]] && kill -0 "${RECON_PID}" 2>/dev/null; then
    echo "   stopping reconstruction_server (PID ${RECON_PID})"
    kill "${RECON_PID}" 2>/dev/null || true
  fi
  # 等子进程实际退出
  for _ in 1 2 3 4 5; do
    if ! { [[ -n "${DET_PID}" ]]   && kill -0 "${DET_PID}"   2>/dev/null; } \
       && ! { [[ -n "${SRV_PID}" ]]   && kill -0 "${SRV_PID}"   2>/dev/null; } \
       && ! { [[ -n "${RECON_PID}" ]] && kill -0 "${RECON_PID}" 2>/dev/null; }; then
      break
    fi
    sleep 0.5
  done
  echo "   日志保留在 ${LOG_DIR}/"
}
trap cleanup EXIT INT TERM

# ── 终端 1：det_pipeline.py ─────────────────────────────────────
echo
echo "▶️  启动 det_pipeline.py (端口 :${DET_PORT})  →  日志: ${DET_LOG}"
python -u det_pipeline.py > "${DET_LOG}" 2>&1 &
DET_PID=$!
echo "   PID = ${DET_PID}"

echo "   等待 OWLv2 + MobileSAM 加载完成 (首次需下载 ~700 MB)..."
DET_TIMEOUT=600   # 最长等 10 分钟（首次下载用）
SECS=0
while ! grep -aqE "Uvicorn running on http://0\.0\.0\.0:${DET_PORT}" "${DET_LOG}" 2>/dev/null; do
  if ! kill -0 "${DET_PID}" 2>/dev/null; then
    echo "❌ det_pipeline 提前退出，最后日志:"
    tail -20 "${DET_LOG}"
    exit 1
  fi
  if (( SECS >= DET_TIMEOUT )); then
    echo "❌ det_pipeline 启动超时 (${DET_TIMEOUT}s)，最后日志:"
    tail -20 "${DET_LOG}"
    exit 1
  fi
  sleep 2
  SECS=$((SECS + 2))
done
echo "✅ det_pipeline 就绪"

# ── reconstruction_server.py (Option 2 sidecar) ─────────────────
echo
echo "▶️  启动 reconstruction_server.py (端口 :${RECON_PORT})  →  日志: ${RECON_LOG}"
echo "   RECONSTRUCTION_PIPELINE_PATH=${RECON_PATH}"
# Pitfall #11: also export legacy SAM3D_PIPELINE_PATH for any downstream code
# that still reads the old name (e.g. ActivePerception_manipulation.streaming_link).
RECONSTRUCTION_PIPELINE_PATH="${RECON_PATH}" SAM3D_PIPELINE_PATH="${RECON_PATH}" \
  python -u reconstruction_server.py > "${RECON_LOG}" 2>&1 &
RECON_PID=$!
echo "   PID = ${RECON_PID}"

echo "   等待 SAM3 模型加载完成 (首次需下载)..."
RECON_TIMEOUT=300   # 最长等 5 分钟
SECS=0
while ! curl -sS http://localhost:${RECON_PORT}/health 2>/dev/null | grep -q '"ok":true'; do
  if ! kill -0 "${RECON_PID}" 2>/dev/null; then
    echo "❌ reconstruction_server 提前退出，最后日志:"
    tail -20 "${RECON_LOG}"
    exit 1
  fi
  if (( SECS >= RECON_TIMEOUT )); then
    echo "❌ reconstruction_server 启动超时 (${RECON_TIMEOUT}s)，最后日志:"
    tail -20 "${RECON_LOG}"
    exit 1
  fi
  sleep 2
  SECS=$((SECS + 2))
done
echo "✅ reconstruction_server 就绪"

# ── 终端 2：action_server.py ────────────────────────────────────
echo
echo "▶️  启动 action_server.py (ZMQ :${ZMQ_PORT})  →  日志: ${SRV_LOG}"
python -u action_server.py > "${SRV_LOG}" 2>&1 &
SRV_PID=$!
echo "   PID = ${SRV_PID}"

echo "   等待 ZMQ bind..."
SRV_TIMEOUT=120
SECS=0
while ! grep -aqE "VLM Action Server listening on tcp://\*:${ZMQ_PORT}|listening on tcp://\*:${ZMQ_PORT}" "${SRV_LOG}" 2>/dev/null; do
  if ! kill -0 "${SRV_PID}" 2>/dev/null; then
    echo "❌ action_server 提前退出，最后日志:"
    tail -20 "${SRV_LOG}"
    exit 1
  fi
  if (( SECS >= SRV_TIMEOUT )); then
    # action_server 可能没打印 listening，但端口已经 bind — 退而求其次检查端口
    if ss -ltn | awk '{print $4}' | grep -qE ":${ZMQ_PORT}$"; then
      echo "✅ action_server 已 bind :${ZMQ_PORT} (没看到 listening 行，但端口在监听)"
      break
    fi
    echo "❌ action_server 启动超时 (${SRV_TIMEOUT}s)，最后日志:"
    tail -20 "${SRV_LOG}"
    exit 1
  fi
  sleep 2
  SECS=$((SECS + 2))
done
[[ ${SECS} -lt ${SRV_TIMEOUT} ]] && echo "✅ action_server 就绪"

# ── 终端 3：client_sapien_<id>.py (前台) ─────────────────────────
echo
echo "▶️  启动 SAPIEN 仿真器 (柜子 ${CABINET_ID})"
echo
echo "   ┌────────────────────────────────────────────────────────┐"
echo "   │  操作:                                                 │"
echo "   │  1) 鼠标点一下 SAPIEN 窗口让它获得焦点                 │"
echo "   │  2) 按 [Space] 触发检测和操作流程                      │"
echo "   │  3) 关窗口或在本终端按 Ctrl+C 退出（会自动清理后端）   │"
echo "   └────────────────────────────────────────────────────────┘"
echo
echo "   实时日志可用另一个终端查看："
echo "     tail -f ${DET_LOG}"
echo "     tail -f ${SRV_LOG}"
echo

python -u "${CLIENT_SCRIPT}"
CLIENT_RC=$?
echo
echo "client_sapien_${CABINET_ID}.py 退出码: ${CLIENT_RC}"
exit ${CLIENT_RC}
