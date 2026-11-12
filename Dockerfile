FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl xz-utils ca-certificates \
    libxi6 libxxf86vm1 libxfixes3 libxrender1 libxkbcommon0 \
    libsm6 libgl1 libegl1 libglu1-mesa \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV BLENDER_URL=https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz

RUN wget -q ${BLENDER_URL} -O /tmp/blender.tar.xz && \
    tar -xf /tmp/blender.tar.xz -C /opt/ && \
    mv /opt/blender-* /opt/blender && \
    ln -s /opt/blender/blender /usr/local/bin/blender && \
    rm /tmp/blender.tar.xz

RUN pip3 install --no-cache-dir runpod

COPY render_worker.py /render_worker.py

CMD ["python3", "-u", "/render_worker.py"]
