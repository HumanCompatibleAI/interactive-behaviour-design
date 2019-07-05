#!/usr/bin/env bash

# Before running, do
#   aws ecr get-login --region us-west-2 --no-include-email
# and run the resulting Docker command to authenticate with the AWS registry

set -e

if ! git status | grep -q 'working tree clean'; then
    echo "Error: working tree not clean; refusing to build" >&2
    exit 1
fi

docker build -t ibd .
docker tag ibd:latest 109526153624.dkr.ecr.us-west-2.amazonaws.com/repository-0:latest
docker push 109526153624.dkr.ecr.us-west-2.amazonaws.com/repository-0:latest