FROM ubuntu:23.10 as builder

# python3.11 is used in ubuntu:23.10 (https://packages.ubuntu.com/jammy/python3)
RUN apt-get update && \
  apt-get install -y \
    python3 \
    python3-pip \
    python3-poetry

WORKDIR /app

# Install split into two steps (the dependencies and the sources)
# in order to leverage the Docker caching
COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry install --no-interaction --no-ansi --no-cache --no-root --no-directory --only main

COPY . .
RUN poetry install --no-interaction --no-ansi --no-cache --only main

FROM ubuntu:23.10 as server

# python3.11 is used in ubuntu:23.10
RUN apt-get update && \
  apt-get install -y python3 adduser

WORKDIR /app

# Copy the sources and virtual env. No poetry.
RUN adduser -u 1001 --disabled-password --gecos "" appuser
COPY --chown=appuser --from=builder /app .

COPY ./scripts/docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh

ENV LOG_LEVEL=INFO
EXPOSE 5000

USER appuser
ENTRYPOINT ["/docker_entrypoint.sh"]

CMD ["uvicorn", "aidial_adapter_vertexai.app:app", "--host", "0.0.0.0", "--port", "5000"]
