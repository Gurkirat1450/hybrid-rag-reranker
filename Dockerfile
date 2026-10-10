# Docker image for the app (free CPU hosting). ONNX runtime via fastembed, no torch.
# The index is NOT built here: run scripts/build_index.py locally and commit index/ first.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run as uid 1000 (required on HF Spaces, harmless elsewhere).
RUN useradd -m -u 1000 user
USER user

# FASTEMBED_CACHE_PATH is read by app/config.py and passed to fastembed as cache_dir by
# prefetch, build_index and the app alike. It's under the user's home, so the RUN below
# creates it owned by (and writable for) uid 1000.
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed \
    ONNX_THREADS=1 \
    TRUST_PROXY=1

WORKDIR /home/user/app

# 1) Dependencies (fastembed brings onnxruntime; no torch).
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

# 2) Bake both models into the image (only config + the model loader are needed, so
#    other code edits keep this layer cached).
COPY --chown=user app/__init__.py app/config.py app/onnx_models.py app/
COPY --chown=user scripts/prefetch_models.py scripts/
RUN python scripts/prefetch_models.py

# 3) App code plus the committed index/ and results/.
COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
