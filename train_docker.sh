#!/usr/bin/env bash
# Start training via Docker, run this from the jibjib-model directory!

docker pull obitech/jibjib-model:latest-gpu
docker run --rm --name jibjib-model -d \
    -v $(pwd)/input:/model/input \
    -v $(pwd)/output:/model/output \
    --runtime=nvidia \
    obitech/jibjib-model:latest-gpu \
    python vggish_train.py \
    --model_version=1.1 \
    --num_mini_batches=1000 \
    --num_batches=101 \
    --save_step=20