# Hugging Face Space (Docker SDK, free CPU).
# The index is NOT built here: run scripts/build_index.py locally and commit index/ first.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# HF Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TRUST_PROXY=1

WORKDIR /home/user/app

# 1) CPU-only torch first, so sentence-transformers does not pull the CUDA build.
RUN pip install --user torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# 2) Everything else (requirements.txt does not list torch).
COPY --chown=user requirements.txt .
RUN pip install --user -r requirements.txt

# 3) Bake both models into the image (only config is needed, so code edits keep this layer cached).
COPY --chown=user app/__init__.py app/config.py app/
COPY --chown=user scripts/prefetch_models.py scripts/
RUN python scripts/prefetch_models.py

# 4) App code plus the committed index/ and results/.
COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
