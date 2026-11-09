FROM python:3.12.12-slim-bookworm

# Generous network settings: PyPI reads from inside the build sometimes time out.
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 PIP_RETRIES=10
WORKDIR /app
# Dependencies first, in their own layer: a code change does not download the wheels again.
# constraints.txt pins every dependency (regenerate: pip freeze inside a fresh image, minus medrag).
# The pip cache is a build cache mount, and the install is retried as a whole: a download that
# stalls or arrives corrupted fails the attempt, and the next one keeps the wheels already fetched.
COPY constraints.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    ok=0; \
    for attempt in 1 2 3 4 5 6 7 8; do \
        if pip install --retries 10 --timeout 100 -r constraints.txt "setuptools>=68"; then ok=1; break; fi; \
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
USER medrag

# The KB .docx files are not in the image: mount them and point MEDRAG_KB_DIR at them.
ENV MEDRAG_KB_DIR=/kb
EXPOSE 8000
CMD ["uvicorn", "medrag.app:create_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
