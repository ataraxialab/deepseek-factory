FROM <BASEIMAGE>

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN mkdir -p /workspace
RUN --mount=type=bind,source=.,target=/mnt /mnt/.layerspec/install.sh

ENV CUCC_PATH=/opt/maca/tools/cu-bridge \
    CUDA_PATH=/opt/maca/tools/cu-bridge 

WORKDIR /workspace
