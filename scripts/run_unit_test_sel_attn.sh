# The used GPU device id
device=0
export PYTHONPATH=$(pwd)

for seqlen in 8192 16384 32768 65536; do
    echo "Running with seqlen=${seqlen}"
    CUDA_VISIBLE_DEVICES=${device} python3 'test/test_FSA_optimized_sel_attn.py' \
        --seqlen ${seqlen} \
        --num-q-heads 32 \
        --num-k-heads 32 \
        --head-dim 128 \
        --block-size 64 \
        --topk 16 \
        --benchmark-iters 20
done
