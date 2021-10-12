import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="face vijnana yolov3",
    version="1.1.0",
    author="Inwoo Chung",
    author_email="gutomitai@gmail.com",
    description="YoloV3 based Keras face detection and identification model.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tonandr/face_vijnana_yolov3",
    packages=setuptools.find_packages(exclude=['analysis', 'samples', 'resource']),
    install_requires=['scipy==1.3.1'
                      , 'pandas==0.25.1'
                      , 'scikit-image==0.15.0'
                      , 'opencv-contrib-python==4.2.0.32'
                      , 'matplotlib==3.1.0'
                      , 'tqdm==4.32.2'
                      , 'ipyparallel'
                      , 'keras==2.2.4'],  # Optional
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)