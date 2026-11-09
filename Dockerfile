# Optional folder of pre-downloaded wheels, supplied as the named build context "wheels" (docker-compose.yml passes
# MEDRAG_WHEELS_DIR; see README). Empty here, so a plain `docker build .` needs no flag: pip then downloads everything.
FROM scratch AS wheels

FROM python:3.12.12-slim-bookworm

# Generous network settings: PyPI reads from inside the build sometimes time out.
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 PIP_RETRIES=10
WORKDIR /app
# Dependencies first, in their own layer: a code change does not download the wheels again.
# constraints.txt pins every dependency (regenerate: pip freeze inside a fresh image, minus medrag).
# The pip cache is a build cache mount, and the install is retried as a whole: a download that
# stalls or arrives corrupted fails the attempt, and the next one keeps the wheels already fetched.
# torch is the CPU-only build (about 190 MB instead of ~2.5 GB with CUDA). Its pin "torch==x+cpu" can only be met
# from the pytorch CPU index or from a wheel in /wheels. pip treats a local wheel and the same wheel on an index as
# equals and may download it, so use_wheelhouse.py first installs the pinned wheels found in /wheels with no network
# at all; the pip install that follows then downloads only what the wheelhouse lacks.
COPY constraints.txt docker/use_wheelhouse.py ./
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=bind,from=wheels,target=/wheels \
    python use_wheelhouse.py constraints.txt /wheels; \
    ok=0; \
    for attempt in 1 2 3 4 5 6 7 8; do \
        if pip install --retries 10 --timeout 100 \
            --extra-index-url https://download.pytorch.org/whl/cpu -r constraints.txt "setuptools>=68"; then ok=1; break; fi; \
        echo "pip install failed (attempt $attempt), retrying"; sleep 5; \
    done; \
    [ "$ok" = 1 ]
COPY pyproject.toml ./
COPY medrag ./medrag
# Build in-tree without isolation (setuptools is already installed above, so nothing is downloaded);
# drop pip's leftovers (build/, *.egg-info) so only the installed package remains.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --timeout 100 --no-deps --no-build-isolation . \
    && rm -rf build medrag.egg-info && useradd --system --uid 10001 medrag
# The Sessions database (SQLite) lives in /var/lib/medrag; compose mounts a named volume on it, which takes over this
# directory's owner the first time it is created. The same goes for /models and /index below.
# BGE-M3's files (about 2.3 GB) are not in the image: `embedder-pull` (python -m medrag.modelstore) puts them in the
# /models volume on the first start; the API reads them from there and keeps its search index in /index.
RUN mkdir -p /var/lib/medrag /models /index && chown medrag /var/lib/medrag /models /index
ENV MEDRAG_SESSIONS_DB=/var/lib/medrag/sessions.db MEDRAG_EMBED_PATH=/models/bge-m3 MEDRAG_INDEX_DIR=/index
USER medrag

# The KB .docx files are not in the image: mount them and point MEDRAG_KB_DIR at them.
ENV MEDRAG_KB_DIR=/kb
EXPOSE 8000
CMD ["uvicorn", "medrag.app:create_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
