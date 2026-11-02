cd /data/lerobot
.venv/bin/lerobot-train \
  --policy.path=/home/apc/.cache/huggingface/lerobot/outputs/train/act_spectrecipher_dataset_new_calib_20261028_085359_2026-10-28_10-05-32/run/checkpoints/010000/pretrained_model \
  --dataset.repo_id=spectrecipher/HIL_fine_tune_datasetlerobot_r2 \
  --dataset.root=/home/apc/demo/datasets/HIL_fine_tune_datasetlerobot_r2 \
  --steps=10000 --batch_size=8 --save_freq=1000 \
  --policy.push_to_hub=false --wandb.enable=false \
  --job_name=act_hil_finetune_r2b_10k \
  --output_dir=/home/apc/demo/outputs/act_hil_finetune_r2b_10k