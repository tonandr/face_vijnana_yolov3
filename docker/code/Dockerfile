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
RUN wget https://www.dropbox.com/s/alr5rkqa3wyfb2w/face_detector.hd5?dl=1 -O face_detector.hd5
RUN chmod 766 train.sh
RUN chmod 766 test.sh
CMD ["/bin/bash"]