#!/usr/bin/env bash
# Host-side: per GPU, per process, number of KFD user queues by type (compute / sdma / ...) and VRAM, from
# /sys/class/kfd/kfd/proc/<pid>/queues/<qid>/{gpuid,type}. Works across containers (host PIDs).
#   ssh <host> 'bash -s' < kfd_queues.sh
declare -A IDX
i=0
for n in $(ls /sys/class/kfd/kfd/topology/nodes | sort -n); do
  g=$(cat /sys/class/kfd/kfd/topology/nodes/$n/gpu_id); [ "$g" = 0 ] && continue
  IDX[$g]=$i; i=$((i+1))
done
declare -A TOT
for p in $(ls /sys/class/kfd/kfd/proc); do
  declare -A CNT=()
  for q in /sys/class/kfd/kfd/proc/$p/queues/*; do
    [ -d "$q" ] || continue
    g=$(cat $q/gpuid 2>/dev/null); t=$(cat $q/type 2>/dev/null)
    CNT["gpu${IDX[$g]} $t"]=$(( ${CNT["gpu${IDX[$g]} $t"]:-0} + 1 ))
    TOT["gpu${IDX[$g]} $t"]=$(( ${TOT["gpu${IDX[$g]} $t"]:-0} + 1 ))
  done
  vram=""
  for f in /sys/class/kfd/kfd/proc/$p/vram_*; do
    g=${f##*_}; v=$(cat $f 2>/dev/null); [ "${v:-0}" -gt 0 ] && vram+=" gpu${IDX[$g]}:$((v/1048576))MB"
  done
  qs=""; for k in "${!CNT[@]}"; do qs+=" [$k x${CNT[$k]}]"; done
  echo "pid $p $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-40) |$vram |$qs"
  unset CNT
done | sort -t'|' -k2
echo "--- totals per GPU"
for k in "${!TOT[@]}"; do echo "$k ${TOT[$k]}"; done | sort
