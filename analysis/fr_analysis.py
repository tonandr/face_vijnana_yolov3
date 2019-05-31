python face_reidentification.py --mode train --raw_data_path /home/ubuntu/nfr/resource --image_size 416 --num_dense1_layers 1 --dense1 64 --num_dense2_layers 0 --dense2 0 --lr 0.0001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 10000 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 0

python face_reidentification.py --mode test --raw_data_path /home/ubuntu/nfr/resource/test --output_file_path /home/ubuntu/nfr/resource/solution.csv --image_size 416 --num_dense1_layers 1 --dense1 64 --num_dense2_layers 0 --dense2 0 --lr 0.0001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 10000 --epochs 2 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --sim_th 40.0 --model_loading 1

python face_detection.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource/training --image_size 416 --num_filters 6 --lr 0.001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 12 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 0
python face_detection.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource/training --image_size 416 --num_filters 6 --lr 0.001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_detection.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource/training --image_size 416 --num_filters 6 --lr 0.0001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_detection.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource/training --image_size 416 --num_filters 6 --lr 0.0001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1

python face_detection.py --mode test --raw_data_path /data/home/ubuntu/nfr/resource/training --output_file_path /data/home/ubuntu/nfr/resource/soluton.csv --image_size 416 --num_filters 6 --lr 0.001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1

python face_reidentification.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource --image_size 416 --num_dense1_layers 1 --dense1 64 --num_dense2_layers 1 --dense2 64 --lr 1e-8 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --step_per_epoch 5000 --epochs 4 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 0
python -m pdb face_reidentification.py --mode test --raw_data_path /data/home/ubuntu/nfr/resource/test --output_file_path /data/home/ubuntu/nfr/resource/solution.csv --image_size 416 --num_dense1_layers 1 --dense1 64 --num_dense2_layers 1 --dense2 64 --lr 1e-8 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --step_per_epoch 5000 --epochs 4 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --entropy_th 40.0 --model_loading 1

python face_detection.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource/training --image_size 416 --num_filters 6 --lr 0.001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 12 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 0
python face_detection.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource/training --image_size 416 --num_filters 6 --lr 0.001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_detection.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource/training --image_size 416 --num_filters 6 --lr 0.0001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_detection.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource/training --image_size 416 --num_filters 6 --lr 0.0001 --beta_1 0.9 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_detection.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource/training --image_size 416 --num_filters 6 --lr 0.0001 --beta_1 0.99 --beta_2 0.9 --decay 0.0 --step_per_epoch 500 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_detection.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource/training --image_size 416 --num_filters 6 --lr 0.0001 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --batch_size 160 --epochs 1 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1 # Multi gpu.
python face_detection.py --mode train --raw_data_path /home/ubuntu/face_recog/resource/training --image_size 416 --num_filters 6 --lr 0.0001 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --batch_size 40 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_detection.py --mode train --raw_data_path /home/ubuntu/face_recog/resource/training --image_size 416 --num_filters 6 --lr 0.0001 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --batch_size 40 --epochs 12 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_detection.py --mode train --raw_data_path /home/ubuntu/face_recog/resource/training --image_size 416 --num_filters 6 --lr 0.0001 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --batch_size 40 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_detection.py --mode train --raw_data_path /home/ubuntu/face_recog/resource/training --image_size 416 --num_filters 6 --lr 0.0001 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --batch_size 40 --epochs 6 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1

python face_reidentification.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource --image_size 416 --num_dense1_layers 1 --dense1 64 --num_dense2_layers 1 --dense2 64 --lr 1e-8 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --step_per_epoch 15000 --epochs 4 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 0

python face_reidentification.py --mode train --raw_data_path /data/home/ubuntu/nfr/resource --image_size 416 --num_dense1_layers 0 --dense1 64 --num_dense2_layers 0 --dense2 0 --lr 0.0001 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --batch_size 40 --epochs 1 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 0

python face_reidentification.py --mode train --raw_data_path /home/ubuntu/face_recog/resource --image_size 416 --num_dense1_layers 0 --dense1 64 --num_dense2_layers 0 --dense2 0 --lr 0.0001 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --batch_size 20 --epochs 1 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1

python face_identification.py --mode train --raw_data_path /home/ubuntu/face_recog/resource --image_size 416 --num_dense1_layers 0 --dense1 64 --num_dense2_layers 0 --dense2 0 --lr 0.0001 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --batch_size 16 --epochs 17 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 0
python face_identification.py --mode train --raw_data_path /home/ubuntu/face_recog/resource --image_size 416 --num_dense1_layers 0 --dense1 64 --num_dense2_layers 0 --dense2 0 --lr 0.00001 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --batch_size 16 --epochs 9 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_identification.py --mode train --raw_data_path /home/ubuntu/face_recog/resource --image_size 416 --num_dense1_layers 0 --dense1 64 --num_dense2_layers 0 --dense2 0 --lr 0.00001 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --batch_size 16 --epochs 3 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_identification.py --mode train --raw_data_path /home/ubuntu/face_recog/resource --image_size 416 --num_dense1_layers 0 --dense1 64 --num_dense2_layers 0 --dense2 0 --lr 0.00001 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --batch_size 16 --epochs 3 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1
python face_identification.py --mode train --raw_data_path /home/ubuntu/face_recog/resource --image_size 416 --num_dense1_layers 0 --dense1 64 --num_dense2_layers 0 --dense2 0 --lr 0.000001 --beta_1 0.99 --beta_2 0.99 --decay 0.0 --batch_size 16 --epochs 3 --face_conf_th 0.5 --nms_iou_th 0.5 --num_cands 60 --model_loading 1

# Face detector training from 2019.5.31
    "fd_conf": {
        "mode": "test",
        "raw_data_path": "D:\\topcoder\\face_recog\\resource",
        "test_path": "D:\\topcoder\\face_recog\\resource\\validation",
        "output_file_path": "solution_fd.csv",
        "multi_gpu": true,
        "num_gpus": 4,
        "yolov3_base_model_load": true,
        "hps": {
            "lr": 0.0001,
            "beta_1": 0.99,
            "beta_2": 0.99,
            "decay": 0.0,
            "epochs": 6,
            "step": 1,
            "batch_size": 40,
            "face_conf_th": 0.5,
            "nms_iou_th": 0.5,
            "num_cands": 60,
            "face_region_ratio_th": 0.8
        },
            
        "nn_arch": {
            "image_size": 416,
            "bb_info_c_size": 6
        },
            
        "model_loading": false
    },



