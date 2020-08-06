#!/usr/bin/env bash

# set -x

ENABLE_CONDA=1
ENABLE_VCAM=1
KILL_PS=1
IS_WORKER=0
IS_CLIENT=0

ARGS=""

while (( "$#" )); do
    case "$1" in
        --no-conda)
            ENABLE_CONDA=0
            shift
            ;;
        --no-vcam)
            ENABLE_VCAM=0
            ARGS="$ARGS --no-stream"
            shift
            ;;
        --keep-ps)
            KILL_PS=0
            shift
            ;;
        --is-worker)
            IS_WORKER=1
            ARGS="$ARGS $1"
            shift
            ;;
        --is-client)
            IS_CLIENT=1
            ARGS="$ARGS $1"
            shift
            ;;
        --is-local-client)
            IS_CLIENT=1
            ARGS="$ARGS --is-client"
            shift
            ;;
        *|-*|--*)
            ARGS="$ARGS $1"
            shift
            ;;
    esac
done

eval set -- "$ARGS"



if [[ $KILL_PS == 1 ]]; then
    kill -9 $(ps aux | grep 'afy/cam_thad.py' | awk '{print $2}') 2> /dev/null
fi

source scripts/settings.sh

if [[ $ENABLE_VCAM == 1 ]]; then
    bash scripts/create_virtual_camera.sh
fi

if [[ $ENABLE_CONDA == 1 ]]; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $CONDA_ENV_NAME
fi

export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/thad

python afy/cam_thad.py \
    --virt-cam $CAMID_VIRT \
    $@
