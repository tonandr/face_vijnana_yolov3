# Docker application of face_detection.py.
# AUthor: Inwoo Chung (gutomitai@gmail.com)
# From Apr 15, 2019.
FROM python:3
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "face_detection.py"]