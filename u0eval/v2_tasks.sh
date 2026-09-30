#!/usr/bin/env bash
# Task groups for the round-2 baseline evaluations (USIM allocation: 40 per goto/grasp/transport,
# 20 per scan/inspect/follow; 700 in total). HALF1 = 340 trials, HALF2 = 360 trials.
LOCO7="goto_charge_station goto_water_tower scan_ship_modern scan_ship_ancient inspect_pipeline_pool inspect_pipeline_sea follow_boat"
LOCO5="scan_ship_modern scan_ship_ancient inspect_pipeline_pool inspect_pipeline_sea follow_boat"
GRASP12="pick_pipe0_shallow pick_pipe1_shallow pick_pipe0_factory pick_pipe1_factory pick_red_shallow pick_redx_shallow pick_red_factory pick_redx_factory pick_blue_shallow pick_bluex_shallow pick_blue_factory pick_bluex_factory"
MANIP13="$GRASP12 transfer_red_shallow"
ALL20="$LOCO7 $MANIP13"
HALF1="$LOCO7 pick_pipe0_shallow pick_pipe1_shallow pick_pipe0_factory pick_pipe1_factory"
HALF2="pick_red_shallow pick_redx_shallow pick_red_factory pick_redx_factory pick_blue_shallow pick_bluex_shallow pick_blue_factory pick_bluex_factory transfer_red_shallow"
# merge_runs <src_root> <dst_root> <dir_glob>: fold another instance's finished blocks into A's eval_runs
merge_runs() {
  local src=$1 dst=$2 pat=$3 d
  for d in "$src"/dataset/eval_runs/$pat; do
    [ -d "$d" ] || continue
    mkdir -p "$dst/dataset/eval_runs/$(basename "$d")"
    rsync -a --exclude 'episode*/' "$d/" "$dst/dataset/eval_runs/$(basename "$d")/"
  done
}
