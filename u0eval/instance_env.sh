#!/usr/bin/env bash
# source this before launching a queue on a shared multi-GPU box:  source instance_env.sh A|B|C|D
#   A: /hy-tmp/u0env   GPU0 :1  ROS 11311+  wam 8000 / percept 8003 / u0 8001 / fallback 8002
#   B: /hy-tmp/u0env_b GPU1 :2  ROS 12311+  wam 8010 / percept 8013 / u0 8011 / fallback 8012
#   C: /hy-tmp/u0env_c GPU0 :3  ROS 13311+  wam 8020 / percept 8023 / u0 8021 / fallback 8022
#   D: /hy-tmp/u0env_d GPU1 :4  ROS 14311+  wam 8030 / percept 8033 / u0 8031 / fallback 8032
# (C/D = second simulator instance on each GPU; a 4090 carries two Stonefish instances.)
case "${1:?A|B|C|D}" in
  A) export U0ENV=/hy-tmp/u0env   DISPLAY_NUM=:1 X_GPU=0 INSTANCE_ROSPORT=11311 PORT_WAM=8000 PORT_WAM_PERCEPT=8003 PORT_U0=8001 PORT_FALLBACK=8002 ;;
  B) export U0ENV=/hy-tmp/u0env_b DISPLAY_NUM=:2 X_GPU=1 INSTANCE_ROSPORT=12311 PORT_WAM=8010 PORT_WAM_PERCEPT=8013 PORT_U0=8011 PORT_FALLBACK=8012 ;;
  C) export U0ENV=/hy-tmp/u0env_c DISPLAY_NUM=:3 X_GPU=0 INSTANCE_ROSPORT=13311 PORT_WAM=8020 PORT_WAM_PERCEPT=8023 PORT_U0=8021 PORT_FALLBACK=8022 ;;
  D) export U0ENV=/hy-tmp/u0env_d DISPLAY_NUM=:4 X_GPU=1 INSTANCE_ROSPORT=14311 PORT_WAM=8030 PORT_WAM_PERCEPT=8033 PORT_U0=8031 PORT_FALLBACK=8032 ;;
  *) echo "instance must be A, B, C or D"; return 1 2>/dev/null || exit 1 ;;
esac
export INSTANCE=$1
# a single-GPU box carries every display on GPU 0 (B/D would otherwise ask nvidia-smi for index 1)
if [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -lt 2 ]; then export X_GPU=0; fi
echo "instance $1: U0ENV=$U0ENV DISPLAY=$DISPLAY_NUM ROS=$INSTANCE_ROSPORT wam=$PORT_WAM percept=$PORT_WAM_PERCEPT"
