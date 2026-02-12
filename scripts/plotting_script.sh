#!/bin/bash

# This script just calls generate_plots.py as in the readme

set -euo pipefail # strict mode
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Project root is $PROJECT_ROOT"

SCRIPTS="scripts"

cd $PROJECT_ROOT
cd $SCRIPTS

python generate_plots.py --config_path config/plotting_config.yml
