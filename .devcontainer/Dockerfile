FROM mcr.microsoft.com/devcontainers/rust:1-1-bookworm

RUN sudo apt-get update && sudo apt-get install -y vim curl

COPY ./build/ ./build/
RUN ./build/setup.sh && rm -r ./build
