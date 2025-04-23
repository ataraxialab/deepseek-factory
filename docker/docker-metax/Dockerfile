FROM mxcr.io/pde-ai-demo/unsloth_maca2.29_py310_torch2.1_image:v1.0

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
COPY requirements.txt.metax /app
RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate unsloth && \
    pip install -r requirements.txt.metax

# Copy the rest of the application into the image
COPY . /app

# Install the deepseekfactory
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate unsloth && \
    pip install .

# Unset http proxy
RUN if [ -n "$HTTP_PROXY" ]; then \
        unset http_proxy; \
        unset https_proxy; \
    fi

# Expose port 7860 for the LLaMA Board
ENV GRADIO_SERVER_PORT 7860
EXPOSE 7860
