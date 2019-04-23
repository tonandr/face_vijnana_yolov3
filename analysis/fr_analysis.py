python face_reidentification.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource --image_size 416 --num_dense1_layers 4 --dense1 64 --num_dense2_layers 4 --dense2 64 --lr 0.001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 1000 --epochs 1 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 0

python face_detection.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource/training --image_size 416 --num_filters 6 --lr 0.001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 12 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 0