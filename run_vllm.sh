#!/bin/bash
# Run vLLM inference in Docker container on Jetson

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
DOCKER_IMAGE="ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin"

docker run --rm --runtime=nvidia --network host \
    -v "$SCRIPT_DIR":/workspace \
    -w /workspace \
    "$DOCKER_IMAGE" \
    python vllm_inference.py "$@"
