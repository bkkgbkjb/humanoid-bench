#!/bin/bash

mkdir -p logdir

for task in h1hand-walk-v0 h1hand-reach-v0 h1hand-hurdle-v0 h1hand-crawl-v0 h1hand-maze-v0 h1hand-push-v0 h1hand-cabinet-v0 h1strong-highbar_hard-v0 h1hand-door-v0 h1hand-truck-v0 h1hand-cube-v0 h1hand-bookshelf_simple-v0 h1hand-bookshelf_hard-v0 h1hand-basketball-v0 h1hand-window-v0 h1hand-spoon-v0 h1hand-kitchen-v0 h1hand-package-v0 h1hand-powerlift-v0 h1hand-room-v0 h1hand-stand-v0 h1hand-run-v0 h1hand-sit_simple-v0 h1hand-sit_hard-v0 h1hand-balance_simple-v0 h1hand-balance_hard-v0 h1hand-stair-v0 h1hand-slide-v0 h1hand-pole-v0 h1hand-insert_normal-v0 h1hand-insert_small-v0; do
  timeout 30s python -m tdmpc2.train disable_wandb=True exp_name=local_test task=humanoid_${task} seed=11111 steps=2000000 &> logdir/local_test_${task}.log
done
