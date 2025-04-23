FROM mxc500-torch2.1-py310:mc2.29.0.9-ubuntu22.04-amd64

#RUN mkdir -p /root/.config/pip/ && echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\n\n[install]\ntrusted-host = pypi.tuna.tsinghua.edu.cn" > /root/.config/pip/pip.conf
#keep base image version equal to env MACA_VERSION value , eg:2.29.0.3
ARG MACA_VERSION=2.29.0.3
ARG PKGS_PATH=/home/metax/yiyu/deepseek-factory

#install vllm 
#disable vllm tn 2 nn
ENV MACA_VLLM_USE_TN_2_NN=0
COPY ./pkgs/mxc500-vllm-${MACA_VERSION} /tmp/mxc500-vllm-${MACA_VERSION}
ENV PATH=$PATH:/opt/conda/bin
#if vllm version less 0.8.2 , should copy the vllm whl package with tn2nn functionality diabled to mxc500-vllm-${MACA_VERSION} dir and delete previous vllm pkg
RUN  cd /tmp/mxc500-vllm-${MACA_VERSION}/wheel && pip3 install *.whl && rm -rf /tmp/mxc500-vllm-${MACA_VERSION}
#install bitsandbytes and update vllm

COPY ./pkgs/*.whl /tmp/
RUN cd /tmp && pip3 install  bitsandbytes-0.45.2*.whl && pip3 uninstall -y flashinfer && rm -rf /tmp/*.whl
#install unsloth
RUN pip install trl==0.14.0 && pip install trl[diffusers] && pip3 install --no-cache-dir --no-deps unsloth==2025.2.4 && pip3 install --no-cache-dir --no-deps unsloth_zoo==2025.2.7 

ENV DEBIAN_FRONTEND=noninteractive

ARG INSTALL_DEEPSPEED=false
#ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG HTTP_PROXY=

# Set the working directory
WORKDIR /app

# Set http proxy
RUN if [ -n "$HTTP_PROXY" ]; then \
        echo "Configuring proxy..."; \
        export http_proxy=$HTTP_PROXY; \
        export https_proxy=$HTTP_PROXY; \
    fi

# Install the requirements
COPY requirements.txt /app
RUN pip install --default-timeout=200 --index-url  http://mirrors.aliyun.com/pypi/simple   -r requirements.txt

# Copy the rest of the application into the image
COPY . /app

# Install the deepseekfactory
RUN pip install . && rm -rf /app/pkgs && pip3 install transformers==4.49.0

#fix bug for sft
#COPY ./metax_unsloth_sft/cross_entropy_loss.py /opt/conda/lib/python3.10/site-packages/unsloth/kernels/
RUN sed -i "326s/.*/                num_warps        = 16,/" /opt/conda/lib/python3.10/site-packages/unsloth/kernels/cross_entropy_loss.py
# Unset http proxy
RUN if [ -n "$HTTP_PROXY" ]; then \
        unset http_proxy; \
        unset https_proxy; \
    fi

# Expose port 7860 for the LLaMA Board
ENV GRADIO_SERVER_PORT 7860
EXPOSE 7860

ENTRYPOINT ["/bin/bash", "-c", "CUDA_VISIBLE_DEVICES=0 GRADIO_SERVER_PORT=8080 deepseekfactory-cli webui"]
