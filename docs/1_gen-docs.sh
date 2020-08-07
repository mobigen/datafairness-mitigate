#!/usr/bin/env bash

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "${CUR_DIR}/.." && pwd)"

sphinx-apidoc -f -o "${CUR_DIR}"/source "${PROJ_DIR}"

make clean
make html
