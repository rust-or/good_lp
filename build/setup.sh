#!/usr/bin/env bash

# Print commands, exit upon error
set -ex

# Change CWD to script location
cd "${0%/*}"

# Install CBC, HiGHS, lp_solve
DEBIAN_FRONTEND=noninteractive sudo apt-get install -y coinor-cbc coinor-libcbc-dev libgsl-dev build-essential cmake clang

# Install CPLEX
if [ -z "$NO_CPLEX" ]
then 
    curl -LO https://github.com/rust-or/good_lp/releases/download/cplex/cplex.bin
    chmod u+x cplex.bin
    ./cplex.bin -f ./response.properties
fi

# Install OR-Tools for CP-SAT
if [ -z "$NO_CPSAT" ]
then
    cd /tmp
    curl -LO 'https://github.com/google/or-tools/releases/download/v9.15/or-tools_amd64_ubuntu-24.04_cpp_v9.15.6755.tar.gz'
    sudo mkdir -p /opt/ortools
    sudo tar -xzf or-tools_amd64_ubuntu-24.04_cpp_v9.15.6755.tar.gz -C /opt/ortools --strip-components=1
    cd /opt/ortools && sudo make test
fi
