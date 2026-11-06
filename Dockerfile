FROM python:3.12.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml ./
COPY medrag ./medrag
RUN pip install . && useradd --system --uid 10001 medrag
USER medrag

# The KB .docx files are not in the image: mount them and point MEDRAG_KB_DIR at them.
ENV MEDRAG_KB_DIR=/kb
EXPOSE 8000
CMD ["uvicorn", "medrag.app:create_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
