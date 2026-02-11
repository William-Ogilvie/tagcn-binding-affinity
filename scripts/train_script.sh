#!/bin/bash

# This script just calls the training_script.py as in the readme

set -euo pipefail # strict mode
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project root is $PROJECT_ROOT"

SCRIPTS="scripts"

cd $PROJECT_ROOT
cd $SCRIPTS

python training_script.py --config_path config/experiments_config.yml --device auto