#!/bin/bash
WANDB_API_KEY=$(cat ./setup/docker/wandb_key)
# git pull

script_and_args="${@:2}"
if [ $1 == "all" ]; then
    gpus="0 1 2 3 4 5 6 7"
else
    gpus=$1
fi

for gpu in $gpus; do
    echo "Launching container jaxrl_$gpu on GPU $gpu"
    docker run \
        --gpus device=$gpu \
        -e WANDB_API_KEY=$WANDB_API_KEY \
        -v $(pwd):/home/duser/rl-jax-template \
        --name jaxrl_$gpu \
        --user $(id -u) \
        --rm \
	-d \
        -t jaxrl \
        /bin/bash -c "$script_and_args"
done
