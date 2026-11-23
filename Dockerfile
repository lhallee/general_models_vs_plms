# syntax=docker/dockerfile:1.7
# general_models_vs_plms: the synthyra_base stack (CUDA 12.8, torch 2.11.0 cu128, transformers 5.17.0) with the
# Anthropic SDK. The code is not copied in: box.sh mounts the shipped tree at run time.
#
# Build: bash box.sh build
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_XET_HIGH_PERFORMANCE=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates python3.12 python3.12-venv && \
    rm -rf /var/lib/apt/lists/* && \
    python3.12 -m venv /opt/venv

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip==26.2.2 && \
    pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install -r /tmp/requirements.txt

RUN python -c "import anthropic, sklearn, torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'anthropic', anthropic.__version__)"

CMD ["bash"]
