#!/usr/bin/env bash
# save as run_and_log_mem.sh，并 chmod +x

export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_DYNAMIC=true
export OMP_NUM_THREADS=1

if [ $# -lt 1 ]; then
  echo "Usage: $0 <your command…>"
  exit 1
fi

LOG_FILE=mem.log
INTERVAL=1   # 采样间隔，单位秒

# 记录脚本启动时刻（秒级时间戳）
START=$(date +%s)

# 启动你的程序（放后台），并捕获它的 PID
"$@" &
PID=$!

# 捕捉 SIGINT/SIGTERM 信号，终止子进程并退出
trap 'echo "Caught SIGINT, killing child $PID…"; kill -TERM $PID 2>/dev/null; exit 1' INT TERM

echo "Started $* with PID=$PID"
echo "# t_elapsed_s RSS_MB" > "$LOG_FILE"

# 轮询记录内存
while kill -0 "$PID" 2>/dev/null; do
  now=$(date +%s)
  t_elapsed=$(( now - START ))

  RSS_KB=$(awk '/^VmRSS:/ {print $2}' /proc/$PID/status 2>/dev/null || echo 0)
  RSS_MB=$(awk -v v="$RSS_KB" 'BEGIN { printf "%.2f\n", v/1024 }')

  echo "$t_elapsed $RSS_MB" >> "$LOG_FILE"
  sleep "$INTERVAL"
done

wait $PID
EXIT=$?

echo "Process $PID exited with code $EXIT"
echo "Memory log in $LOG_FILE"
exit $EXIT
