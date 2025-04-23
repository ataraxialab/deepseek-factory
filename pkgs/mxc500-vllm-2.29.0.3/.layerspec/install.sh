#!/bin/bash
set -e
CURDIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

if command -v apt-get >/dev/null; then
    apt-get update && apt-get install -y --no-install-recommends \
    vim zip wget tar tzdata \
    build-essential libssl-dev libxml2 openssh-server libxau6 \
    libopenblas-dev \
    libibverbs1 librdmacm1 ibverbs-providers libibumad3 libibverbs-dev librdmacm-dev ibverbs-utils libibumad-dev \
    && apt-get clean
elif command -v yum >/dev/null; then
    yum install -y vim zip wget tar tzdata \
    make cmake ninja-build gcc gcc-c++ libxml2 openssh-server libXau \
    openblas-devel \
    libibverbs librdmacm libibumad libibverbs-utils librdmacm-utils \
    && yum clean all 
else
    echo "Could not determine the package manager type."
fi

ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

mkdir -p /var/run/sshd

. /etc/profile.d/conda.sh

pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
pip config set install.trusted-host mirrors.aliyun.com

pip install tokenizers==0.20.3
pip install $CURDIR/../wheel/*.whl
pip install orjson==3.10.6
pip install ninja psutil fastapi uvicorn[standard] aioprometheus[starlette] sentencepiece numpy 'transformers>=4.36.0' fastapi pydantic==2.7.4 outlines==0.0.44 httpx