#FROM python:3-alpine3.20
FROM python:3.12-slim

# RUN apk update && apk add --no-cache cmake make gcc g++ musl-dev openssl-dev linux-headers suitesparse-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libsuitesparse-dev && rm -rf /var/lib/apt/lists/*

COPY . /opt/p3s
WORKDIR /opt/p3s

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir ./p3s/cpp
RUN python3 -m pip install --no-cache-dir .[test]

ENTRYPOINT ["python3", "-m", "pytest", "tests"]
