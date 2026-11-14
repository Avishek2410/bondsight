# Wire-bond inspector: one image for the Raspberry Pi 5 (arm64) and normal PCs (amd64).
# Build from the repo root:  docker build -f docker/Dockerfile -t wirebond-inspector .
FROM python:3.11-slim

# libgl/libglib: needed by OpenCV for USB / network cameras
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY synth/ synth/
COPY inspector/ inspector/
# slow factory / home networks: wait longer for downloads and retry instead of failing the build
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
RUN pip install --no-cache-dir ./synth ./inspector

# run as an ordinary user, not root; /data holds the database + pictures
RUN useradd --create-home --uid 1000 inspector && mkdir /data && chown inspector /data
USER inspector

EXPOSE 8080
# 0.0.0.0 INSIDE the container is fine: compose publishes the port on 127.0.0.1 only
CMD ["wbinspect", "--config", "/config/config.yaml", "--host", "0.0.0.0"]
