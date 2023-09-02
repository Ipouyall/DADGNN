NGRAM ?= 4
NAME ?= model
BAR ?= 1
WD ?= 1e-6
DROPOUT ?= 0.5
NUM_HIDDEN ?= 64
NUM_LAYERS ?= 5
NUM_HEADS ?= 2
K ?= 5
ALPHA ?= 0.5
DATASET ?= OLIDv1
EDGES ?= 1
RAND ?= 42


all:
    @python train_pre.py \
        --ngram=${NGRAM} \
        --name=${NAME} \
        --bar=${BAR} \
        --wd=${WD} \
        --dropout=${DROPOUT} \
        --num_hidden=${NUM_HIDDEN} \
        --num_layers=${NUM_LAYERS} \
        --num_heads=${NUM_HEADS} \
        --k=${K} \
        --alpha=${ALPHA} \
        --dataset=${DATASET} \
        --edges=${EDGES} \
        --rand=${RAND}
