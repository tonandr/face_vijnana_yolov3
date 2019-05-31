# Docker application of face re-identification.
# AUthor: Inwoo Chung (gutomitai@gmail.com)
# From Apr 24, 2019.
FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /usr/src/app
COPY . .
RUN apt-get update
RUN apt-get -y install wget
RUN apt-get -y install libsm6 libxrender1 libfontconfig1 libxext6
RUN pip install --no-cache-dir -r requirements.txt
RUN wget https://pjreddie.com/media/files/yolov3.weights -O yolov3.weights
RUN wget https://www.dropbox.com/s/3nizyylif4s9kr7/face_detector.hd5?dl=1 -O face_detector.hd5
RUN wget https://www.dropbox.com/s/j414cgv62e8lq1t/yolov3_base.hd5?dl=1 -O yolov3_base.hd5
RUN wget https://www.dropbox.com/s/blyf39a4qxmqiyd/db_facial_id.pobj?dl=1 -O db_facial_id.pobj
RUN wget https://www.dropbox.com/s/ia2jibf0utza76n/face_reidentifier.hd5?dl=1 -O face_reidentifier.hd5
RUN chmod 766 train_fr.sh
RUN chmod 766 test_fr.sh
RUN chmod 766 train.sh
RUN chmod 766 test.sh
CMD ["/bin/bash"]