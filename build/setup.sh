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
