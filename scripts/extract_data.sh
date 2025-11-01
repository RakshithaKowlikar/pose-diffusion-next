#!/bin/bash
set -e

IN_DIR="downloads"
OUT_DIR="data/raw"
mkdir -p "$OUT_DIR"

for file in CMU.tar.bz2 KIT.tar.bz2 Transitions.tar.bz2; do
    echo "Extracting $file"
    tar -xjf "$IN_DIR/$file" -C "$OUT_DIR"
done

echo "Extracted dataset to $OUT_DIR"