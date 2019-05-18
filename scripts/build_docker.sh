#!/usr/bin/env bash

# Before running, do
#   aws ecr get-login --region us-west-2 --no-include-email
# and run the resulting Docker command to authenticate with the AWS registry

set -e

mujoco_key_url=$1
if [[ $mujoco_key_url == "" ]]; then
    echo "Usage: $0 <MuJoCo key URL>" >&2
    exit 1
fi

if ! git status | grep -q 'working tree clean'; then
    echo "Error: working tree not clean; refusing to build" >&2
    exit 1
fi

docker build -t ibd . --build-arg MUJOCO_KEY=$mujoco_key_url
docker tag ibd:latest 181772678123.dkr.ecr.us-west-2.amazonaws.com/batch-images:latest
docker push 181772678123.dkr.ecr.us-west-2.amazonaws.com/batch-images:latest
