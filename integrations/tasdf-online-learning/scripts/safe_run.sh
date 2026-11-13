#!/usr/bin/env bash
# Run a heavy command (pytest, learning curve, training) without overheating the laptop.
#
#   scripts/safe_run.sh <command...>
#   SAFE_RUN_CONTAINER=tasdf-online scripts/safe_run.sh docker exec tasdf-online python -m scripts.learning_curve
#
# - Refuses to start off AC power.
# - Runs the command with nice 10 and 4 BLAS threads.
# - Every EVERY seconds reads GPU (nvidia-smi) and CPU (x86_pkg_temp) temperature. At GPU >= GPU_HOT or
#   CPU >= CPU_HOT, or when AC is lost, it pauses the command (SIGSTOP on its process group, or `docker pause`
#   when SAFE_RUN_CONTAINER is set) and resumes once both are below COOL and AC is back. Nothing is lost.
# - Logs temperatures to SAFE_RUN_LOG, so a sudden power-off leaves evidence.
set -u

GPU_HOT=${GPU_HOT:-80}
CPU_HOT=${CPU_HOT:-90}
COOL=${COOL:-70}
EVERY=${EVERY:-5}
CONTAINER=${SAFE_RUN_CONTAINER:-}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
LOG=${SAFE_RUN_LOG:-$ROOT/.data/tasdf-online/reports/safe_run.log}
mkdir -p "$(dirname "$LOG")"

[ $# -gt 0 ] || { echo "usage: $0 <command...>" >&2; exit 2; }

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG" >&2; }

on_ac() {
    local f
    for f in /sys/class/power_supply/*/online; do
        [ "$(cat "$f" 2>/dev/null)" = 1 ] && return 0
    done
    return 1
}

gpu_temp() {
    local t
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
    echo "${t:-0}"
}

cpu_temp() {
    local z
    for z in /sys/class/thermal/thermal_zone*; do
        if [ "$(cat "$z/type" 2>/dev/null)" = x86_pkg_temp ]; then
            echo $(( $(cat "$z/temp") / 1000 ))
            return
        fi
    done
    echo 0
}

if ! on_ac; then
    log "refused: not on AC power, plug in the charger first: $*"
    exit 3
fi

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 setsid nice -n 10 "$@" &
pid=$!
paused=0

pause_cmd() {
    if [ -n "$CONTAINER" ]; then docker pause "$CONTAINER" >/dev/null; else kill -STOP -- "-$pid"; fi
}

resume_cmd() {
    if [ -n "$CONTAINER" ]; then docker unpause "$CONTAINER" >/dev/null; else kill -CONT -- "-$pid"; fi
}

stop_all() {
    [ "$paused" = 1 ] && resume_cmd
    kill -TERM -- "-$pid" 2>/dev/null
    log "interrupted, stopped pid $pid"
    # shortcut: killing `docker exec` does not stop the process inside the container
    [ -n "$CONTAINER" ] && log "still running in $CONTAINER without the watchdog: docker exec $CONTAINER pkill -f <script>"
    exit 130
}
trap stop_all INT TERM

log "start pid=$pid gpu=$(gpu_temp)C cpu=$(cpu_temp)C: $*"
tick=0
while kill -0 "$pid" 2>/dev/null; do
    g=$(gpu_temp); c=$(cpu_temp); ac=1
    on_ac || ac=0
    if [ "$paused" = 0 ] && { [ "$g" -ge "$GPU_HOT" ] || [ "$c" -ge "$CPU_HOT" ] || [ "$ac" = 0 ]; }; then
        pause_cmd; paused=1
        log "pause gpu=${g}C cpu=${c}C ac=$ac"
    elif [ "$paused" = 1 ] && [ "$g" -lt "$COOL" ] && [ "$c" -lt "$COOL" ] && [ "$ac" = 1 ]; then
        resume_cmd; paused=0
        log "resume gpu=${g}C cpu=${c}C"
    elif [ $(( tick % 12 )) = 0 ]; then
        log "temp gpu=${g}C cpu=${c}C ac=$ac paused=$paused"
    fi
    tick=$(( tick + 1 ))
    sleep "$EVERY"
done
wait "$pid"
rc=$?
log "done rc=$rc gpu=$(gpu_temp)C cpu=$(cpu_temp)C"
exit "$rc"
