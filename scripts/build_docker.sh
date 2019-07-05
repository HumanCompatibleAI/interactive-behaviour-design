#!/usr/bin/env bash

set -e

if ! git status | grep -q 'working tree clean'; then
    echo "Error: working tree not clean; refusing to build" >&2
    exit 1
fi

docker build -t ibd .
docker push mrahtz/ibd
