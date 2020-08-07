#!/usr/bin/env bash

CUR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sphinx-quickstart -q --sep -p DataFairness-Mitigation -a 'Seungwu Baek, Mobigen' "${CUR_DIR}"

CONF_TXT="

--- Path setup ---
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.abspath('.'), '..', '..')))

--- General configuration ---
extensions = [
    'sphinxcontrib.napoleon'
]
"

echo "Before you go, make sure 'conda install sphinxcontrib.napoleon'"

echo "Insert below to source/conf.py"
echo "${CONF_TXT}"
