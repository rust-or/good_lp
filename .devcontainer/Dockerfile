FROM mcr.microsoft.com/devcontainers/rust:1-1-bookworm

RUN apt-get update && apt-get install -y vim curl

COPY ./build/ ./build/
RUN ./build/setup.sh && rm -r ./build
