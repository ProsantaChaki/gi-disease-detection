FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_TIMEOUT=600
ENV PIP_RETRIES=5

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install in small batches to avoid timeout on large downloads
RUN pip install --no-cache-dir "tensorboard>=2.13.0"

RUN pip install --no-cache-dir \
    "albumentations>=1.3.0" \
    "opencv-python>=4.8.0" \
    "Pillow>=10.0.0" \
    "scikit-image>=0.21.0"

RUN pip install --no-cache-dir "pyiqa>=0.1.7"

RUN pip install --no-cache-dir \
    "numpy>=1.24.0" \
    "pandas>=2.0.0" \
    "scikit-learn>=1.3.0" \
    "scipy>=1.11.0"

RUN pip install --no-cache-dir \
    "matplotlib>=3.7.0" \
    "plotly>=5.14.0" \
    "seaborn>=0.12.0"

RUN pip install --no-cache-dir \
    "python-dotenv>=1.0.0" \
    "PyYAML>=6.0" \
    "tqdm>=4.65.0"

RUN pip install --no-cache-dir \
    "ipywidgets>=8.0.0" \
    "jupyter>=1.0.0" \
    "notebook>=7.0.0"

EXPOSE 8888 6006