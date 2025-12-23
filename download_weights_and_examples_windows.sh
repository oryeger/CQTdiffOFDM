#!/usr/bin/env bash
set -e

echo "Downloading weights..."
curl -L -o cqt_weights.pt https://github.com/eloimoliner/CQTdiff/releases/download/weights_and_examples/cqt_weights.pt

mkdir -p experiments/cqt
mv cqt_weights.pt experiments/cqt/

echo "Downloading examples..."
curl -L -o examples.zip https://github.com/eloimoliner/CQTdiff/releases/download/weights_and_examples/examples.zip

echo "Unzipping examples..."
tar -xf examples.zip

echo "Done."
