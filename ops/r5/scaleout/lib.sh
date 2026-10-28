# Shared helpers for the r5 scale-out stage scripts (sourced, not executed).
# Every function is safe under `set -u`; nothing here starts work on its own.

GCS_ROOT="${GCS_ROOT:-gs://anthropocene-rl-datasets-508912/model9/r5}"
SCALEOUT_GCS="${SCALEOUT_GCS:-$GCS_ROOT/scaleout}"
ACTOR_NORMALIZER_SHA256="3c8c5ca300bab11d9651dfc4bfbaa36c7bae08e4c0236773c84e2237f1fa57de"

# The r5 collection batches (fit8-b03r replaces fit8-b03; heldout50 is never a fit input).
FIT_BATCHES_ALL="fit8-b01 fit8-b02 fit8-b03r fit8-b04 fit8-b05 fit8-b06 fit8-b07 fit8-b08"
CAL_BATCHES_ALL="cal8-b01 cal8-b02 cal8-b03"

ts() { date -u +%FT%TZ; }
log() { echo "$(ts) [${STAGE_NAME:-scaleout}] $*" | tee -a "${LOG_FILE:-/dev/null}" >&2; }
die() { log "FAIL $*"; exit 1; }

# join_paths DIR NAME... -> "DIR/NAME1 DIR/NAME2 ..."
join_paths() { local d=$1; shift; local out=() n; for n in "$@"; do out+=("$d/$n"); done; echo "${out[*]}"; }

mem_available_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
disk_free_gb() { df -Pk "$1" | awk 'NR==2{print int($4/1048576)}'; }

require_file() { [ -e "$1" ] || die "missing $1${2:+ ($2)}"; }

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

# gcs_put SRC DEST: copy one file (never fails the stage; logged).
gcs_put() {
  timeout 600 gcloud storage cp --no-user-output-enabled "$1" "$2" < /dev/null \
    || log "upload failed: $1 -> $2"
}
# gcs_sync SRC_DIR DEST_DIR [EXCLUDE_REGEX]: incremental upload (never fails the stage).
gcs_sync() {
  local args=(-r)
  [ -n "${3:-}" ] && args+=(-x "$3")
  timeout 3600 gcloud storage rsync --no-user-output-enabled "${args[@]}" "$1" "$2" < /dev/null \
    || log "sync failed: $1 -> $2"
}
# gcs_fetch_dir SRC DEST: incremental download of one published dataset directory; a
# .fetched marker skips the listing on later runs (the loaders verify shard hashes anyway).
gcs_fetch_dir() {
  local src=$1 dest=$2
  if [ -f "$dest/dataset.json" ] && [ -f "$dest/.fetched" ]; then return 0; fi
  mkdir -p "$dest"
  timeout 7200 gcloud storage rsync -r --no-user-output-enabled "$src" "$dest" < /dev/null \
    || die "download failed: $src"
  [ -f "$dest/dataset.json" ] || die "no dataset.json in $src"
  touch "$dest/.fetched"
}

# memwatch_start OUT_CSV [INTERVAL_S]: background sampler of host memory, python RSS
# (sum and largest process), the .NET bridges' summed RSS and GPU memory when present.
# Prints the sampler PID.
memwatch_start() {
  local out=$1 every=${2:-60}
  (
    [ -s "$out" ] || echo "utc,mem_available_gb,mem_total_gb,python_rss_sum_gb,python_rss_max_gb,dotnet_rss_sum_gb,gpu_mem_used_mib" > "$out"
    while true; do
      mem=$(awk '/MemAvailable/{a=$2} /MemTotal/{t=$2} END{printf "%.1f,%.1f", a/1048576, t/1048576}' /proc/meminfo)
      py=$(ps -C python,python3,python3.12 -o rss= 2>/dev/null | awk '{s+=$1; if($1>m)m=$1} END{printf "%.2f,%.2f", s/1048576, m/1048576}')
      dn=$(ps -C dotnet -o rss= 2>/dev/null | awk '{s+=$1} END{printf "%.2f", s/1048576}')
      gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      echo "$(ts),$mem,$py,$dn,${gpu:-}" >> "$out"
      sleep "$every"
    done
  ) > /dev/null 2>&1 < /dev/null &
  echo $!
}

# memwatch_peak CSV -> "peak_python_rss_gb=... min_mem_available_gb=..."
memwatch_peak() {
  awk -F, 'NR>1{if($5>m)m=$5; if(a==""||$2<a)a=$2} END{printf "peak_python_rss_gb=%s min_mem_available_gb=%s\n", m, a}' "$1" 2>/dev/null
}

# shutdown_vm: the stage scripts end the VM unless NO_SHUTDOWN=1 (interactive debugging).
shutdown_vm() {
  if [ "${NO_SHUTDOWN:-0}" = "1" ]; then
    log "NO_SHUTDOWN=1: leaving the VM running"
    return 0
  fi
  log "shutting the VM down"
  sync
  sudo -n shutdown -h now "r5 scale-out stage finished" || log "shutdown failed (sudo)"
}
