#!/bin/bash
# Face detector training script.
# Author: Inwoo Chung (gutomitai@gmail.com)

TRAINING_DATA_PATH="$1"

python face_detection.py --mode train --raw_data_path $TRAINING_DATA_PAT --image_size 416 --num_filters 6 --lr 0.001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 12 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 0
python face_detection.py --mode train --raw_data_path $TRAINING_DATA_PAT --image_size 416 --num_filters 6 --lr 0.001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_detection.py --mode train --raw_data_path $TRAINING_DATA_PAT --image_size 416 --num_filters 6 --lr 0.0001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_detection.py --mode train --raw_data_path $TRAINING_DATA_PAT --image_size 416 --num_filters 6 --lr 0.0001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1