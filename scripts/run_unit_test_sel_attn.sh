# The used GPU device id
device=4
export PYTHONPATH=$(pwd)

for seqlen in 32768; do
    echo "Running with seqlen=${seqlen}"
    CUDA_VISIBLE_DEVICES=${device} python3 'test/test_FSA_optimized_sel_attn.py' \
        --seqlen ${seqlen} \
        --num-q-heads 32 \
        --num-k-heads 32 \
        --head-dim 128 \
        --block-size 64 \
        --topk 16 \
        --benchmark-iters 2
done
