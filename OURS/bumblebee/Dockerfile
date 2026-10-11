ARG BASE=openbumblebee-audit-deps:latest
FROM ${BASE}
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends iproute2 iputils-ping && rm -rf /var/lib/apt/lists/*
