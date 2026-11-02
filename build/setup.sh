#!/usr/bin/env bash

# Print commands, exit upon error
set -ex

# Change CWD to script location
cd "${0%/*}"

# Install CBC, HiGHS, lp_solve
sudo apt-get update
DEBIAN_FRONTEND=noninteractive sudo apt-get install -y coinor-cbc coinor-libcbc-dev libgsl-dev build-essential cmake clang

# Install CPLEX
if [ -z "$NO_CPLEX" ]
then 
    CPLEX_INSTALLER=cplex-22.2.0-linux-x86-64.bin
    curl --fail --location --retry 3 --output "$CPLEX_INSTALLER" \
        "https://github.com/rust-or/good_lp/releases/download/cplex/$CPLEX_INSTALLER"
    echo "7b93d12a6747695dd13edb03b815400660f3a51e3d68823d6933adc686b6c278  $CPLEX_INSTALLER" | sha256sum --check
    chmod u+x "$CPLEX_INSTALLER"
    "./$CPLEX_INSTALLER" -f ./response.properties
fi
