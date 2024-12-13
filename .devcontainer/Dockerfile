FROM mcr.microsoft.com/devcontainers/rust:1-1-bookworm

WORKDIR /workspace

RUN sudo apt-get update && sudo apt-get install -y vim curl \
    ## coin_cbc
    coinor-cbc coinor-libcbc-dev \
    ## C(++) build chain (HiGHS, lp_solve)
    build-essential
