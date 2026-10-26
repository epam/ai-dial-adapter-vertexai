FROM python:3.11-alpine3.20 AS builder

RUN apk update && apk upgrade --no-cache libcrypto3 libssl3
# geos-dev is required for https://pypi.org/project/shapely/2.0.5/
RUN apk add --no-cache alpine-sdk linux-headers geos-dev
RUN pip install poetry==2.1.1

WORKDIR /app

# Install split into two steps (the dependencies and the sources)
# in order to leverage the Docker caching
COPY pyproject.toml poetry.lock poetry.toml ./
RUN poetry install --no-interaction --no-ansi --no-cache --no-root --no-directory --only main

COPY . .
RUN poetry install --no-interaction --no-ansi --no-cache --only main

# pip is not needed at runtime; its vendored deps (msgpack, urllib3, setuptools) carry CVEs
RUN .venv/bin/python -m pip uninstall -y pip

FROM python:3.11-alpine3.20 AS server

RUN apk update && apk upgrade --no-cache libcrypto3 libssl3
# fix CVE-2023-52425
RUN apk upgrade --no-cache libexpat
# fix CVE-2026-23949 and GHSA-xmj3-9gf7-h52w
RUN pip install setuptools==84.1.0
# fix CVE-2026-24049
RUN pip install wheel==0.46.2
# pip is not needed at runtime; removing it fixes all pip CVEs, incl. those in its
# vendored copies of msgpack, urllib3 and pkg_resources (setuptools 70.3.0, not the
# standalone setuptools installed above). Must stay after the last pip usage.
RUN pip uninstall -y pip
# fix CVE-2025-6965 and CVE-2026-22184 and CVE-2026-22184
RUN apk upgrade --no-cache sqlite-libs zlib musl musl-utils
# fix CVE-2024-58251 and CVE-2025-46394 (busybox) and CVE-2026-34743 (xz-libs)
RUN apk upgrade --no-cache busybox busybox-binsh ssl_client xz-libs

WORKDIR /app

# Copy the sources and virtual env. No poetry.
RUN adduser -u 1001 --disabled-password --gecos "" appuser
COPY --chown=appuser --from=builder /app .

COPY ./scripts/docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh

ENV LOG_LEVEL=INFO
ENV PYDANTIC_V2=1

EXPOSE 5000

USER appuser

HEALTHCHECK  --interval=10s --timeout=5s --start-period=30s --retries=6 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:5000/health || exit 1

ENTRYPOINT ["/docker_entrypoint.sh"]
