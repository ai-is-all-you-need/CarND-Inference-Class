FROM tensorflow/tensorflow:1.3.0-gpu
RUN apt-get update && apt-get install -y libxrender1 libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
RUN pip install ipython opencv-python