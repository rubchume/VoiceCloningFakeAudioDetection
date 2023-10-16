#!/bin/bash

# Extract conda dependencies
conda_deps=$(awk '/dependencies:/,/pip:/' conda.yaml | grep -vE '(dependencies:|pip:|- pip)')

# Extract pip dependencies
pip_deps=$(awk '/pip:/{flag=1; next} /[^-]/{flag=0} flag' conda.yaml | sed 's/^- //')

# Install conda dependencies
if [ ! -z "$conda_deps" ]; then
    conda install --yes $conda_deps
fi

# Install pip dependencies
if [ ! -z "$pip_deps" ]; then
    pip install $pip_deps
fi
