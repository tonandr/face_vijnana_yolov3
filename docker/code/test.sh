#!/bin/bash
# Face detector test script.
# Author: Inwoo Chung (gutomitai@gmail.com)

TEST_DATA_PATH="$1"
OUTPUT_FILE_PATH="$2"

python face_detection.py --mode test --raw_data_path $TEST_DATA_PATH --output_file_path $OUTPUT_FILE_PATH --image_size 416 --num_filters 6 --step_per_epoch 500 --epochs 1 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
