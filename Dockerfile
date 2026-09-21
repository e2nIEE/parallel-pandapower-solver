# SPDX-FileCopyrightText: 2026 Fraunhofer IEE
#
# SPDX-License-Identifier: BSD-3-Clause
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libsuitesparse-dev && rm -rf /var/lib/apt/lists/*

COPY . /opt/p3s
WORKDIR /opt/p3s

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir ./p3s/cpp
RUN python3 -m pip install --no-cache-dir .[test]

ENTRYPOINT ["python3", "-m", "pytest", "tests"]
