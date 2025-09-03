#include <cuda_runtime.h>
#include <cuda_bf16.h>   // <-- ADDED for bfloat16 support
#include <torch/extension.h>
#include <cmath>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>


// Define constants for the kernel.
constexpr int BLOCK_DIM_X = 128;

// __device__ functions for math, equivalent to tl.exp2 and tl.log2
__device__ inline float exp2_fast(float x) {
    return exp2f(x);
}

__device__ inline float log2_fast(float x) {
    return log2f(x);
}

#define WARP_SIZE 32

// Helper to determine padding for shared memory to avoid bank conflicts
// For int/float, typically 32 banks. If size is a multiple of 32, add 1.
#define SMEM_PAD_INT_OR_FLOAT ((WARP_SIZE % 32 == 0) ? 1 : 0)

template<int BLOCK_SIZE_D, int TOPK>
__global__ void lse_reduce_kernel_cuda(
    // Pointers
    float* lse_ptr,
    const float* m_ij_ptr,
    const float* l_ij_first_ptr,
    const float* l_ij_rest_ptr,
    const float* m_ij_last_ptr,
    const int* t_ptr,
    const int* token_index_mapping_ptr,
    // Scalars
    int start_head_id,
    int total_len,
    // Strides
    int64_t stride_lse_n,
    int64_t stride_m_ij_b, int64_t stride_m_ij_n,
    int64_t stride_l_ij_fb, int64_t stride_l_ij_fn,
    int64_t stride_l_ij_rb, int64_t stride_l_ij_rn,
    int64_t stride_tn, int64_t stride_tk,
    int64_t stride_tim_b, int64_t stride_tim_n
) {
    // const int tid = threadIdx.x; // 0 to WARP_SIZE-1
    // const int pid_q_j = blockIdx.x; // Each block handles one token
    const int tid = threadIdx.x;
    const int pid_q_j = blockIdx.x * blockDim.x + tid;

    if (pid_q_j >= total_len) {
        return;
    }

    // ===================================================================================
    // SHARED MEMORY SETUP
    // ===================================================================================
    extern __shared__ char smem_storage[]; // Dynamically sized shared memory

    // Calculate offsets for different shared memory regions
    // All sizes are per-thread * padded.
    size_t offset_t_shared = 0;
    size_t offset_real_token_index_shared = offset_t_shared + sizeof(int) * WARP_SIZE * (TOPK + SMEM_PAD_INT_OR_FLOAT);

    // Two stages for pipelined data
    size_t offset_m_ij_stage0 = offset_real_token_index_shared + sizeof(int) * WARP_SIZE * (TOPK + SMEM_PAD_INT_OR_FLOAT);
    size_t offset_m_ij_stage1 = offset_m_ij_stage0 + sizeof(float) * WARP_SIZE; // One float per thread per stage

    size_t offset_l_ij_stage0 = offset_m_ij_stage1 + sizeof(float) * WARP_SIZE;
    size_t offset_l_ij_stage1 = offset_l_ij_stage0 + sizeof(float) * WARP_SIZE;

    // BLOCK_SIZE_D elements for o_tiles. Each thread loads BLOCK_SIZE_D / WARP_SIZE.
    // Total size is BLOCK_SIZE_D * sizeof(__nv_bfloat16) * 2 stages
    // We need to ensure o_tiles_sh_stage0/1 are aligned for bf16 access.
    // Assuming BLOCK_SIZE_D is a multiple of WARP_SIZE for simplicity.
    // Padding here is for the entire BLOCK_SIZE_D if needed, not per-thread
    // size_t offset_o_tiles_stage0 = offset_l_ij_stage1 + sizeof(float) * WARP_SIZE;
    // size_t offset_o_tiles_stage1 = offset_o_tiles_stage0 + sizeof(__nv_bfloat16) * BLOCK_SIZE_D;

    // size_t offset_acc_o_scales_stage0 = offset_o_tiles_stage1 + sizeof(__nv_bfloat16) * BLOCK_SIZE_D;
    // size_t offset_acc_o_scales_stage1 = offset_acc_o_scales_stage0 + sizeof(float) * WARP_SIZE;

    // Pointers to shared memory regions
    int* t_shared = (int*)(smem_storage + offset_t_shared);
    int* real_token_index_shared = (int*)(smem_storage + offset_real_token_index_shared);

    float* m_ij_sh_stage0 = (float*)(smem_storage + offset_m_ij_stage0);
    float* m_ij_sh_stage1 = (float*)(smem_storage + offset_m_ij_stage1);
    float* l_ij_sh_stage0 = (float*)(smem_storage + offset_l_ij_stage0);
    float* l_ij_sh_stage1 = (float*)(smem_storage + offset_l_ij_stage1);
    // __nv_bfloat16* o_tiles_sh_stage0 = (__nv_bfloat16*)(smem_storage + offset_o_tiles_stage0);
    // __nv_bfloat16* o_tiles_sh_stage1 = (__nv_bfloat16*)(smem_storage + offset_o_tiles_stage1); // Fixed here
    // float* acc_o_scales_sh_stage0 = (float*)(smem_storage + offset_acc_o_scales_stage0);
    // float* acc_o_scales_sh_stage1 = (float*)(smem_storage + offset_acc_o_scales_stage1);


    // ===================================================================================
    // INITIAL LOAD: 't' and 'real_token_index' (not pipelined, loaded all at once)
    // ===================================================================================
    const int* t_global_ptr = t_ptr + pid_q_j * stride_tn;
    // Load all 't' and derived 'real_token_index' values for this token into shared memory
    for (int k = 0; k < TOPK; ++k) {
        // Load 't'
        int current_t = t_global_ptr[k * stride_tk];
        t_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + k] = current_t;

        // Load 'real_token_index' only if 't' is valid
        if (current_t != -1) {
            real_token_index_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + k] = token_index_mapping_ptr[current_t * stride_tim_b + pid_q_j * stride_tim_n];
        } else {
            // Set a safe default if t is -1 to avoid invalid memory access later
            real_token_index_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + k] = 0; // Or any sentinel value
        }
    }
    __syncthreads(); // Ensure all 't' and 'real_token_index' are loaded

    // ===================================================================================
    // INITIALIZATION: Load initial 'o' and 'lse' state
    // ===================================================================================

    float lse = lse_ptr[pid_q_j * stride_lse_n];
    const float m_ij_last = m_ij_last_ptr[pid_q_j];

    // ===================================================================================
    // SOFTWARE PIPELINING LOOP
    // ===================================================================================
    int current_stage = 0;
    int next_stage = 1;

    // Prefetch for the first iteration (block_id = 0)
    // This part runs while the main loop would normally be doing computation
    // but there's no prior computation to overlap with.
    {
        int t_pref = t_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + 0];
        int real_token_index_pref = real_token_index_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + 0];

        // Pointers for current prefetch stage
        float* m_ij_sh_curr_pref = (current_stage == 0) ? m_ij_sh_stage0 : m_ij_sh_stage1;
        float* l_ij_sh_curr_pref = (current_stage == 0) ? l_ij_sh_stage0 : l_ij_sh_stage1;

        if (t_pref != -1) {
            // Determine global pointers based on t_pref and real_token_index_pref
            int real_block_pos_pref;
            const float* l_ij_ptr_pref;
            int64_t stride_l_ij_b_pref, stride_l_ij_n_pref;

            if (t_pref == 0) {
                real_block_pos_pref = 0;
                l_ij_ptr_pref = l_ij_first_ptr;

                stride_l_ij_b_pref = stride_l_ij_fb; stride_l_ij_n_pref = stride_l_ij_fn;

            } else {
                real_block_pos_pref = t_pref - 1;
                l_ij_ptr_pref = l_ij_rest_ptr;

                stride_l_ij_b_pref = stride_l_ij_rb; stride_l_ij_n_pref = stride_l_ij_rn;

            }

            // Load scalars for this thread and store to shared memory
            m_ij_sh_curr_pref[tid] = m_ij_ptr[t_pref * stride_m_ij_b + pid_q_j * stride_m_ij_n];
            l_ij_sh_curr_pref[tid] = l_ij_ptr_pref[real_block_pos_pref * stride_l_ij_b_pref + real_token_index_pref * stride_l_ij_n_pref];


        } else {
            // If t_pref is -1, fill shared memory with safe defaults
            m_ij_sh_curr_pref[tid] = 0.0f;
            l_ij_sh_curr_pref[tid] = 0.0f;
        }
    } // End of initial prefetch block

    // Main pipelined loop
    for (int block_id = 0; block_id < TOPK; ++block_id) {
        __syncthreads(); // Sync after prefetch/before compute

        // Pointers for current compute stage (using data that was prefetched)
        float* m_ij_sh_curr_comp = (current_stage == 0) ? m_ij_sh_stage0 : m_ij_sh_stage1;
        float* l_ij_sh_curr_comp = (current_stage == 0) ? l_ij_sh_stage0 : l_ij_sh_stage1;

        // Get 't' from shared memory for this block_id
        const int t = t_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + block_id];

        if (t != -1) {
            // Get data for current compute iteration from shared memory
            const float m_ij = m_ij_sh_curr_comp[tid];
            const float l_ij = l_ij_sh_curr_comp[tid];

            // --- LSE Update ---
            const float delta = lse - m_ij;
            const float log_delta = exp2_fast(delta) + l_ij;
            lse = m_ij + log2_fast(log_delta);

        }

        // Prefetch data for the NEXT iteration (block_id + 1)
        if (block_id + 1 < TOPK) {
            int t_next = t_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + block_id + 1];
            int real_token_index_next = real_token_index_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + block_id + 1];

            // Pointers for next prefetch stage
            float* m_ij_sh_next_pref = (next_stage == 0) ? m_ij_sh_stage0 : m_ij_sh_stage1;
            float* l_ij_sh_next_pref = (next_stage == 0) ? l_ij_sh_stage0 : l_ij_sh_stage1;

            if (t_next != -1) {
                // Determine global pointers based on t_next and real_token_index_next
                int real_block_pos_next;
                const float* l_ij_ptr_next;
                int64_t stride_l_ij_b_next, stride_l_ij_n_next;

                if (t_next == 0) {
                    real_block_pos_next = 0;
                    l_ij_ptr_next = l_ij_first_ptr;
                    stride_l_ij_b_next = stride_l_ij_fb; stride_l_ij_n_next = stride_l_ij_fn;

                } else {
                    real_block_pos_next = t_next - 1;
                    l_ij_ptr_next = l_ij_rest_ptr;

                    stride_l_ij_b_next = stride_l_ij_rb; stride_l_ij_n_next = stride_l_ij_rn;

                }

                // Load scalars for this thread and store to shared memory
                m_ij_sh_next_pref[tid] = m_ij_ptr[t_next * stride_m_ij_b + pid_q_j * stride_m_ij_n];
                l_ij_sh_next_pref[tid] = l_ij_ptr_next[real_block_pos_next * stride_l_ij_b_next + real_token_index_next * stride_l_ij_n_next];

            } else {
                // If t_next is -1, fill shared memory with safe defaults
                m_ij_sh_next_pref[tid] = 0.0f;
                l_ij_sh_next_pref[tid] = 0.0f;
            }
        }

        // Swap stages for the next iteration
        current_stage = 1 - current_stage;
        next_stage = 1 - next_stage;
    }
    __syncthreads(); // Ensure last iteration's computation is complete

    // ===================================================================================
    // FINALIZATION: Apply final scale and store results
    // ===================================================================================

    lse_ptr[pid_q_j * stride_lse_n] = lse;
}


template<int BLOCK_SIZE_D, int TOPK>
__global__ void o_reduce_kernel_cuda(
    // Pointers
    float* lse_ptr,
    const float* m_ij_last_ptr,
    __nv_bfloat16* o_ptr,
    const __nv_bfloat16* o_tiles_first_ptr,
    const __nv_bfloat16* o_tiles_rest_ptr,
    const float* acc_o_scales_first_ptr,
    const float* acc_o_scales_rest_ptr,
    const int* t_ptr,
    const int* token_index_mapping_ptr,
    // Scalars
    int start_head_id,
    int total_len,
    // Strides
    int64_t stride_lse_n,
    int64_t stride_on, int64_t stride_od,
    int64_t stride_otfb, int64_t stride_otfn, int64_t stride_otfd,
    int64_t stride_otrb, int64_t stride_otrn, int64_t stride_otrd,
    int64_t stride_acc_fb, int64_t stride_acc_fn,
    int64_t stride_acc_rb, int64_t stride_acc_rn,
    int64_t stride_tn, int64_t stride_tk,
    int64_t stride_tim_b, int64_t stride_tim_n
) {
    // ===================================================================================
    // THREAD AND BLOCK CONFIGURATION
    // ===================================================================================
    constexpr int WARP_SIZE = 32;
    constexpr int THREADS_PER_TOKEN = 32;
    constexpr int TOKENS_PER_BLOCK = 4;  // 128 threads / 32 threads per token
    constexpr int VALUES_PER_THREAD = BLOCK_SIZE_D / WARP_SIZE; // 128/32 = 4
    
    const int tid = threadIdx.x;
    const int token_local_id = tid / THREADS_PER_TOKEN; // 0-3, which token within block
    const int thread_in_token = tid % THREADS_PER_TOKEN; // 0-31, thread within token
    
    const int base_token_id = blockIdx.x * TOKENS_PER_BLOCK;
    
    // ===================================================================================
    // SHARED MEMORY SETUP FOR DOUBLE BUFFERING
    // ===================================================================================
    extern __shared__ char smem_storage[];
    
    size_t offset = 0;
    
    // Static data (loaded once per token)
    size_t offset_t_shared = offset;
    offset += sizeof(int) * TOKENS_PER_BLOCK * TOPK;
    
    size_t offset_real_token_index_shared = offset;
    offset += sizeof(int) * TOKENS_PER_BLOCK * TOPK;
    
    // Double buffered data - Stage 0
    size_t offset_acc_o_scales_stage0 = offset;
    offset += sizeof(float) * TOKENS_PER_BLOCK;
    
    size_t offset_o_tiles_stage0 = offset;
    offset += sizeof(float) * TOKENS_PER_BLOCK * BLOCK_SIZE_D; // Use float in shared memory
    
    // Double buffered data - Stage 1
    size_t offset_acc_o_scales_stage1 = offset;
    offset += sizeof(float) * TOKENS_PER_BLOCK;
    
    size_t offset_o_tiles_stage1 = offset;
    offset += sizeof(float) * TOKENS_PER_BLOCK * BLOCK_SIZE_D;
    
    // Cast shared memory pointers
    int* t_shared = (int*)(smem_storage + offset_t_shared);
    int* real_token_index_shared = (int*)(smem_storage + offset_real_token_index_shared);
    
    float* acc_o_scales_sh_stage0 = (float*)(smem_storage + offset_acc_o_scales_stage0);
    float* acc_o_scales_sh_stage1 = (float*)(smem_storage + offset_acc_o_scales_stage1);
    float* o_tiles_sh_stage0 = (float*)(smem_storage + offset_o_tiles_stage0);
    float* o_tiles_sh_stage1 = (float*)(smem_storage + offset_o_tiles_stage1);
    
    // ===================================================================================
    // LOOP OVER TOKENS IN THIS BLOCK
    // ===================================================================================
    for (int token_offset = 0; token_offset < TOKENS_PER_BLOCK; ++token_offset) {
        const int token_id = base_token_id + token_offset;
        
        // Early exit if token_id exceeds total_len
        if (token_id >= total_len) {
            break;
        }
        
        // Only threads assigned to this token participate
        if (token_local_id == token_offset) {
            
            // ===================================================================================
            // STATIC DATA LOADING: Load 't' and 'real_token_index' once
            // ===================================================================================
            const int* t_global_ptr = t_ptr + token_id * stride_tn;
            
            // Thread 0 loads all t values for this token
            if (thread_in_token == 0) {
                for (int k = 0; k < TOPK; ++k) {
                    int current_t = t_global_ptr[k * stride_tk];
                    t_shared[token_offset * TOPK + k] = current_t;
                    
                    if (current_t != -1) {
                        real_token_index_shared[token_offset * TOPK + k] = 
                            token_index_mapping_ptr[current_t * stride_tim_b + token_id * stride_tim_n];
                    } else {
                        real_token_index_shared[token_offset * TOPK + k] = 0;
                    }
                }
            }
            
            // ===================================================================================
            // INITIALIZATION: Load initial 'o' state and constants
            // ===================================================================================
            float acc_o[VALUES_PER_THREAD];
            __nv_bfloat16* o_local_ptr = o_ptr + token_id * stride_on;
            
            // Each thread loads its portion of BLOCK_SIZE_D
            for (int d_idx = 0; d_idx < VALUES_PER_THREAD; ++d_idx) {
                int d = thread_in_token * VALUES_PER_THREAD + d_idx;
                if (d < BLOCK_SIZE_D) {
                    acc_o[d_idx] = __bfloat162float(o_local_ptr[d * stride_od]);
                } else {
                    acc_o[d_idx] = 0.0f;
                }
            }
            
            // Thread 0 loads constants (other threads will read from shared memory later)
            __shared__ float shared_lse[TOKENS_PER_BLOCK];
            __shared__ float shared_m_ij_last[TOKENS_PER_BLOCK];
            __shared__ float shared_final_scale[TOKENS_PER_BLOCK];
            
            if (thread_in_token == 0) {
                float lse = lse_ptr[token_id * stride_lse_n];
                float m_ij_last = m_ij_last_ptr[token_id];
                
                shared_lse[token_offset] = lse;
                shared_m_ij_last[token_offset] = m_ij_last;
                shared_final_scale[token_offset] = exp2f(m_ij_last - lse);
            }
            
            __syncthreads(); // Wait for static data loading
            
            // ===================================================================================
            // SOFTWARE PIPELINING LOOP WITH DOUBLE BUFFERING
            // ===================================================================================
            int current_stage = 0;
            
            // Prefetch for the first iteration (block_id = 0)
            if (TOPK > 0) {
                int t_pref = t_shared[token_offset * TOPK + 0];
                int real_token_index_pref = real_token_index_shared[token_offset * TOPK + 0];
                
                // Choose stage 0 for initial prefetch
                float* acc_o_scales_sh_curr = (current_stage == 0) ? acc_o_scales_sh_stage0 : acc_o_scales_sh_stage1;
                float* o_tiles_sh_curr = (current_stage == 0) ? o_tiles_sh_stage0 : o_tiles_sh_stage1;
                
                if (t_pref != -1) {
                    // Determine pointers and strides
                    int real_block_pos_pref;
                    const __nv_bfloat16* o_tiles_ptr_pref;
                    const float* acc_o_scales_ptr_pref;
                    int64_t stride_otb_pref, stride_otn_pref, stride_otd_pref;
                    int64_t stride_acc_b_pref, stride_acc_n_pref;
                    
                    if (t_pref == 0) {
                        real_block_pos_pref = 0;
                        o_tiles_ptr_pref = o_tiles_first_ptr;
                        acc_o_scales_ptr_pref = acc_o_scales_first_ptr;
                        stride_otb_pref = stride_otfb; stride_otn_pref = stride_otfn; stride_otd_pref = stride_otfd;
                        stride_acc_b_pref = stride_acc_fb; stride_acc_n_pref = stride_acc_fn;
                    } else {
                        real_block_pos_pref = t_pref - 1;
                        o_tiles_ptr_pref = o_tiles_rest_ptr;
                        acc_o_scales_ptr_pref = acc_o_scales_rest_ptr;
                        stride_otb_pref = stride_otrb; stride_otn_pref = stride_otrn; stride_otd_pref = stride_otrd;
                        stride_acc_b_pref = stride_acc_rb; stride_acc_n_pref = stride_acc_rn;
                    }
                    
                    // Thread 0 loads scalar
                    if (thread_in_token == 0) {
                        acc_o_scales_sh_curr[token_offset] = acc_o_scales_ptr_pref[real_block_pos_pref * stride_acc_b_pref + real_token_index_pref * stride_acc_n_pref];
                    }
                    
                    // All threads cooperatively load o_tiles
                    const __nv_bfloat16* o_tiles_local_ptr_pref = o_tiles_ptr_pref + 
                        real_block_pos_pref * stride_otb_pref + real_token_index_pref * stride_otn_pref;
                    
                    for (int d_idx = 0; d_idx < VALUES_PER_THREAD; ++d_idx) {
                        int d = thread_in_token * VALUES_PER_THREAD + d_idx;
                        if (d < BLOCK_SIZE_D) {
                            // Convert bf16 to float when loading into shared memory
                            o_tiles_sh_curr[token_offset * BLOCK_SIZE_D + d] = 
                                __bfloat162float(o_tiles_local_ptr_pref[d * stride_otd_pref]);
                        }
                    }
                } else {
                    // If t_pref is -1, fill with safe defaults
                    if (thread_in_token == 0) {
                        acc_o_scales_sh_curr[token_offset] = 0.0f;
                    }
                    
                    for (int d_idx = 0; d_idx < VALUES_PER_THREAD; ++d_idx) {
                        int d = thread_in_token * VALUES_PER_THREAD + d_idx;
                        if (d < BLOCK_SIZE_D) {
                            o_tiles_sh_curr[token_offset * BLOCK_SIZE_D + d] = 0.0f;
                        }
                    }
                }
            }
            
            // Main pipelined loop
            for (int block_id = 0; block_id < TOPK; ++block_id) {
                __syncthreads(); // Sync after prefetch/before compute
                
                // Choose current compute stage
                float* acc_o_scales_sh_comp = (current_stage == 0) ? acc_o_scales_sh_stage0 : acc_o_scales_sh_stage1;
                float* o_tiles_sh_comp = (current_stage == 0) ? o_tiles_sh_stage0 : o_tiles_sh_stage1;
                
                // Get 't' from shared memory for this block_id
                const int t = t_shared[token_offset * TOPK + block_id];
                
                if (t != -1) {
                    // ===================================================================================
                    // COMPUTE: Update accumulator with current tile
                    // ===================================================================================
                    const float acc_o_scale_tile = acc_o_scales_sh_comp[token_offset];
                    
                    for (int d_idx = 0; d_idx < VALUES_PER_THREAD; ++d_idx) {
                        int d = thread_in_token * VALUES_PER_THREAD + d_idx;
                        if (d < BLOCK_SIZE_D) {
                            float o_tile_val = o_tiles_sh_comp[token_offset * BLOCK_SIZE_D + d];
                            // Update accumulator: o_tile + acc_o * scale
                            acc_o[d_idx] = o_tile_val + acc_o[d_idx] * acc_o_scale_tile;
                        }
                    }
                }
                
                // ===================================================================================
                // PREFETCH: Load data for next iteration (if exists)
                // ===================================================================================
                if (block_id + 1 < TOPK) {
                    int next_stage = 1 - current_stage;
                    int t_next = t_shared[token_offset * TOPK + (block_id + 1)];
                    int real_token_index_next = real_token_index_shared[token_offset * TOPK + (block_id + 1)];
                    
                    // Choose next stage for prefetch
                    float* acc_o_scales_sh_next = (next_stage == 0) ? acc_o_scales_sh_stage0 : acc_o_scales_sh_stage1;
                    float* o_tiles_sh_next = (next_stage == 0) ? o_tiles_sh_stage0 : o_tiles_sh_stage1;
                    
                    if (t_next != -1) {
                        // Determine pointers and strides for next iteration
                        int real_block_pos_next;
                        const __nv_bfloat16* o_tiles_ptr_next;
                        const float* acc_o_scales_ptr_next;
                        int64_t stride_otb_next, stride_otn_next, stride_otd_next;
                        int64_t stride_acc_b_next, stride_acc_n_next;
                        
                        if (t_next == 0) {
                            real_block_pos_next = 0;
                            o_tiles_ptr_next = o_tiles_first_ptr;
                            acc_o_scales_ptr_next = acc_o_scales_first_ptr;
                            stride_otb_next = stride_otfb; stride_otn_next = stride_otfn; stride_otd_next = stride_otfd;
                            stride_acc_b_next = stride_acc_fb; stride_acc_n_next = stride_acc_fn;
                        } else {
                            real_block_pos_next = t_next - 1;
                            o_tiles_ptr_next = o_tiles_rest_ptr;
                            acc_o_scales_ptr_next = acc_o_scales_rest_ptr;
                            stride_otb_next = stride_otrb; stride_otn_next = stride_otrn; stride_otd_next = stride_otrd;
                            stride_acc_b_next = stride_acc_rb; stride_acc_n_next = stride_acc_rn;
                        }
                        
                        // Thread 0 loads scalar for next iteration
                        if (thread_in_token == 0) {
                            acc_o_scales_sh_next[token_offset] = acc_o_scales_ptr_next[real_block_pos_next * stride_acc_b_next + real_token_index_next * stride_acc_n_next];
                        }
                        
                        // All threads cooperatively load o_tiles for next iteration
                        const __nv_bfloat16* o_tiles_local_ptr_next = o_tiles_ptr_next + 
                            real_block_pos_next * stride_otb_next + real_token_index_next * stride_otn_next;
                        
                        for (int d_idx = 0; d_idx < VALUES_PER_THREAD; ++d_idx) {
                            int d = thread_in_token * VALUES_PER_THREAD + d_idx;
                            if (d < BLOCK_SIZE_D) {
                                // Convert bf16 to float when loading into shared memory
                                o_tiles_sh_next[token_offset * BLOCK_SIZE_D + d] = 
                                    __bfloat162float(o_tiles_local_ptr_next[d * stride_otd_next]);
                            }
                        }
                    } else {
                        // If t_next is -1, fill with safe defaults
                        if (thread_in_token == 0) {
                            acc_o_scales_sh_next[token_offset] = 0.0f;
                        }
                        
                        for (int d_idx = 0; d_idx < VALUES_PER_THREAD; ++d_idx) {
                            int d = thread_in_token * VALUES_PER_THREAD + d_idx;
                            if (d < BLOCK_SIZE_D) {
                                o_tiles_sh_next[token_offset * BLOCK_SIZE_D + d] = 0.0f;
                            }
                        }
                    }
                }
                
                // Flip stage for next iteration
                current_stage = 1 - current_stage;
            } // End of main pipelined loop
            
            // ===================================================================================
            // FINALIZATION: Apply final scale and write back results
            // ===================================================================================
            const float final_scale = shared_final_scale[token_offset];
            
            // Apply final scaling and write back to global memory
            for (int d_idx = 0; d_idx < VALUES_PER_THREAD; ++d_idx) {
                int d = thread_in_token * VALUES_PER_THREAD + d_idx;
                if (d < BLOCK_SIZE_D) {
                    acc_o[d_idx] *= final_scale;
                    o_local_ptr[d * stride_od] = __float2bfloat16_rn(acc_o[d_idx]);
                }
            }
            
        } // End of token_local_id == token_offset check
        
        __syncthreads(); // Sync between different tokens in the block
        
    } // End of token loop
}

// // Naive
// template<int BLOCK_SIZE_D, int TOPK>
// __global__ void o_reduce_kernel_cuda(
//     // Pointers
//     float* lse_ptr,
//     const float* m_ij_last_ptr,
//     __nv_bfloat16* o_ptr,                  // <-- CORRECTED TYPE
//     const __nv_bfloat16* o_tiles_first_ptr,  // <-- CORRECTED TYPE
//     const __nv_bfloat16* o_tiles_rest_ptr,   // <-- CORRECTED TYPE
//     const float* acc_o_scales_first_ptr,
//     const float* acc_o_scales_rest_ptr,
//     const int* t_ptr,
//     const int* token_index_mapping_ptr,
//     // Scalars
//     int start_head_id,
//     int total_len,
//     // Strides
//     int64_t stride_lse_n,
//     int64_t stride_on, int64_t stride_od,
//     int64_t stride_otfb, int64_t stride_otfn, int64_t stride_otfd,
//     int64_t stride_otrb, int64_t stride_otrn, int64_t stride_otrd,
//     int64_t stride_acc_fb, int64_t stride_acc_fn,
//     int64_t stride_acc_rb, int64_t stride_acc_rn,
//     int64_t stride_tn, int64_t stride_tk,
//     int64_t stride_tim_b, int64_t stride_tim_n
// ) {
//     // ===================================================================================
//     // KERNEL SETUP: Map threads to tokens
//     // ===================================================================================
//     const int tid = threadIdx.x;
//     const int pid_q_j = blockIdx.x * blockDim.x + tid;

//     if (pid_q_j >= total_len) {
//         return;
//     }

//     // ===================================================================================
//     // OPTIMIZATION: Bulk load 't' indices into shared memory
//     // ===================================================================================
//     extern __shared__ int t_shared_storage[]; // Dynamically sized shared memory
//     int* t_shared = t_shared_storage; // [BLOCK_DIM_X][TOPK]

//     const int* t_global_ptr = t_ptr + pid_q_j * stride_tn;

//     // Each thread loads its own TOPK indices into its row in shared memory.
//     // This can be further optimized for coalescing, but is a correct starting point.
//     for (int k = 0; k < TOPK; ++k) {
//         t_shared[tid * TOPK + k] = t_global_ptr[k * stride_tk];
        
//     }
//     __syncthreads(); // Ensure all 't' indices are loaded before proceeding.

//     // ===================================================================================
//     // INITIALIZATION: Load initial 'o' and 'lse' state
//     // ===================================================================================
//     // Accumulator for 'o' must be in high precision (float32)
//     float acc_o[BLOCK_SIZE_D];

//     // Load initial 'o' values, converting from bf16 to float32
//     __nv_bfloat16* o_local_ptr = o_ptr + pid_q_j * stride_on;
//     for (int d = 0; d < BLOCK_SIZE_D; ++d) {
//         if (d * stride_od < total_len * stride_on) { // Boundary check
//             acc_o[d] = __bfloat162float(o_local_ptr[d * stride_od]);
//         } else {
//             acc_o[d] = 0.0f;
//         }
//     }

//     float lse = lse_ptr[pid_q_j * stride_lse_n];
//     const float m_ij_last = m_ij_last_ptr[pid_q_j];

//     // ===================================================================================
//     // MAIN REDUCTION LOOP
//     // ===================================================================================
//     for (int block_id = 0; block_id < TOPK; ++block_id) {
//         const int t = t_shared[tid * TOPK + block_id];

//         if (t != -1) {
//             // Triton's branching logic translated to CUDA
//             int real_block_pos;
            
//             const __nv_bfloat16* o_tiles_ptr;
//             const float* acc_o_scales_ptr;
            
//             int64_t stride_otb, stride_otn;
//             int64_t stride_acc_b, stride_acc_n;

//             if (t == 0) {
//                 real_block_pos = 0;
                
//                 o_tiles_ptr = o_tiles_first_ptr;
//                 acc_o_scales_ptr = acc_o_scales_first_ptr;
                
//                 stride_otb = stride_otfb; stride_otn = stride_otfn;
//                 stride_acc_b = stride_acc_fb; stride_acc_n = stride_acc_fn;
//             } else {
//                 real_block_pos = t - 1;
                
//                 o_tiles_ptr = o_tiles_rest_ptr;
//                 acc_o_scales_ptr = acc_o_scales_rest_ptr;
                
//                 stride_otb = stride_otrb; stride_otn = stride_otrn;
//                 stride_acc_b = stride_acc_rb; stride_acc_n = stride_acc_rn;
//             }

//             const int real_token_index = token_index_mapping_ptr[t * stride_tim_b + pid_q_j * stride_tim_n];

//             // --- Vector loads and 'o' update ---
//             const __nv_bfloat16* o_tiles_local_ptr = o_tiles_ptr + real_block_pos * stride_otb + real_token_index * stride_otn;
//             const float acc_o_scale_tile = acc_o_scales_ptr[real_block_pos * stride_acc_b + real_token_index * stride_acc_n];

//             // This loop loads o_tiles (bf16), converts to float, and updates acc_o (float)
//             for (int d = 0; d < BLOCK_SIZE_D; ++d) {
//                 float o_tile_val = __bfloat162float(o_tiles_local_ptr[d * stride_otfd]);
//                 acc_o[d] = o_tile_val + acc_o[d] * acc_o_scale_tile;
//             }
//         }
//     }

//     // ===================================================================================
//     // FINALIZATION: Apply final scale and store results
//     // ===================================================================================
//     const float final_scale = exp2_fast(m_ij_last - lse);

//     // Store final 'o', converting from float32 accumulator to bf16 storage
//     for (int d = 0; d < BLOCK_SIZE_D; ++d) {
//          if (d * stride_od < total_len * stride_on) { // Boundary check
//             acc_o[d] *= final_scale;
//             o_local_ptr[d * stride_od] = __float2bfloat16_rn(acc_o[d]);
//         }
//     }

// }



// // Define constants for the kernel.
// constexpr int BLOCK_DIM_X = 128;

// // __device__ functions for math, equivalent to tl.exp2 and tl.log2
// __device__ inline float exp2_fast(float x) {
//     return exp2f(x);
// }

// __device__ inline float log2_fast(float x) {
//     return log2f(x);
// }

// #define WARP_SIZE 32

// // Helper to determine padding for shared memory to avoid bank conflicts
// // For int/float, typically 32 banks. If size is a multiple of 32, add 1.
// #define SMEM_PAD_INT_OR_FLOAT ((WARP_SIZE % 32 == 0) ? 1 : 0)

// template<int BLOCK_SIZE_D, int TOPK>
// __global__ void reduce_kernel_cuda(
//     // Pointers
//     float* lse_ptr,
//     const float* m_ij_ptr,
//     const float* l_ij_first_ptr,
//     const float* l_ij_rest_ptr,
//     const float* m_ij_last_ptr,
//     __nv_bfloat16* o_ptr,
//     const __nv_bfloat16* o_tiles_first_ptr,
//     const __nv_bfloat16* o_tiles_rest_ptr,
//     const float* acc_o_scales_first_ptr,
//     const float* acc_o_scales_rest_ptr,
//     const int* t_ptr,
//     const int* token_index_mapping_ptr,
//     // Scalars
//     int start_head_id,
//     int total_len,
//     // Strides
//     int64_t stride_lse_n,
//     int64_t stride_m_ij_b, int64_t stride_m_ij_n,
//     int64_t stride_l_ij_fb, int64_t stride_l_ij_fn,
//     int64_t stride_l_ij_rb, int64_t stride_l_ij_rn,
//     int64_t stride_on, int64_t stride_od,
//     int64_t stride_otfb, int64_t stride_otfn, int64_t stride_otfd,
//     int64_t stride_otrb, int64_t stride_otrn, int64_t stride_otrd,
//     int64_t stride_acc_fb, int64_t stride_acc_fn,
//     int64_t stride_acc_rb, int64_t stride_acc_rn,
//     int64_t stride_tn, int64_t stride_tk,
//     int64_t stride_tim_b, int64_t stride_tim_n
// ) {
//     // const int tid = threadIdx.x; // 0 to WARP_SIZE-1
//     // const int pid_q_j = blockIdx.x; // Each block handles one token
//     const int tid = threadIdx.x;
//     const int pid_q_j = blockIdx.x * blockDim.x + tid;

//     if (pid_q_j >= total_len) {
//         return;
//     }

//     // ===================================================================================
//     // SHARED MEMORY SETUP
//     // ===================================================================================
//     extern __shared__ char smem_storage[]; // Dynamically sized shared memory

//     // Calculate offsets for different shared memory regions
//     // All sizes are per-thread * padded.
//     size_t offset_t_shared = 0;
//     size_t offset_real_token_index_shared = offset_t_shared + sizeof(int) * WARP_SIZE * (TOPK + SMEM_PAD_INT_OR_FLOAT);

//     // Two stages for pipelined data
//     size_t offset_m_ij_stage0 = offset_real_token_index_shared + sizeof(int) * WARP_SIZE * (TOPK + SMEM_PAD_INT_OR_FLOAT);
//     size_t offset_m_ij_stage1 = offset_m_ij_stage0 + sizeof(float) * WARP_SIZE; // One float per thread per stage

//     size_t offset_l_ij_stage0 = offset_m_ij_stage1 + sizeof(float) * WARP_SIZE;
//     size_t offset_l_ij_stage1 = offset_l_ij_stage0 + sizeof(float) * WARP_SIZE;

//     // BLOCK_SIZE_D elements for o_tiles. Each thread loads BLOCK_SIZE_D / WARP_SIZE.
//     // Total size is BLOCK_SIZE_D * sizeof(__nv_bfloat16) * 2 stages
//     // We need to ensure o_tiles_sh_stage0/1 are aligned for bf16 access.
//     // Assuming BLOCK_SIZE_D is a multiple of WARP_SIZE for simplicity.
//     // Padding here is for the entire BLOCK_SIZE_D if needed, not per-thread
//     size_t offset_o_tiles_stage0 = offset_l_ij_stage1 + sizeof(float) * WARP_SIZE;
//     size_t offset_o_tiles_stage1 = offset_o_tiles_stage0 + sizeof(__nv_bfloat16) * BLOCK_SIZE_D;

//     size_t offset_acc_o_scales_stage0 = offset_o_tiles_stage1 + sizeof(__nv_bfloat16) * BLOCK_SIZE_D;
//     size_t offset_acc_o_scales_stage1 = offset_acc_o_scales_stage0 + sizeof(float) * WARP_SIZE;

//     // Pointers to shared memory regions
//     int* t_shared = (int*)(smem_storage + offset_t_shared);
//     int* real_token_index_shared = (int*)(smem_storage + offset_real_token_index_shared);

//     float* m_ij_sh_stage0 = (float*)(smem_storage + offset_m_ij_stage0);
//     float* m_ij_sh_stage1 = (float*)(smem_storage + offset_m_ij_stage1);
//     float* l_ij_sh_stage0 = (float*)(smem_storage + offset_l_ij_stage0);
//     float* l_ij_sh_stage1 = (float*)(smem_storage + offset_l_ij_stage1);
//     __nv_bfloat16* o_tiles_sh_stage0 = (__nv_bfloat16*)(smem_storage + offset_o_tiles_stage0);
//     __nv_bfloat16* o_tiles_sh_stage1 = (__nv_bfloat16*)(smem_storage + offset_o_tiles_stage1); // Fixed here
//     float* acc_o_scales_sh_stage0 = (float*)(smem_storage + offset_acc_o_scales_stage0);
//     float* acc_o_scales_sh_stage1 = (float*)(smem_storage + offset_acc_o_scales_stage1);


//     // ===================================================================================
//     // INITIAL LOAD: 't' and 'real_token_index' (not pipelined, loaded all at once)
//     // ===================================================================================
//     const int* t_global_ptr = t_ptr + pid_q_j * stride_tn;
//     // Load all 't' and derived 'real_token_index' values for this token into shared memory
//     for (int k = 0; k < TOPK; ++k) {
//         // Load 't'
//         int current_t = t_global_ptr[k * stride_tk];
//         t_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + k] = current_t;

//         // Load 'real_token_index' only if 't' is valid
//         if (current_t != -1) {
//             real_token_index_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + k] = token_index_mapping_ptr[current_t * stride_tim_b + pid_q_j * stride_tim_n];
//         } else {
//             // Set a safe default if t is -1 to avoid invalid memory access later
//             real_token_index_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + k] = 0; // Or any sentinel value
//         }
//     }
//     __syncthreads(); // Ensure all 't' and 'real_token_index' are loaded

//     // ===================================================================================
//     // INITIALIZATION: Load initial 'o' and 'lse' state
//     // ===================================================================================
//     // Each thread handles a portion of BLOCK_SIZE_D
//     constexpr int TILE_SIZE_D = BLOCK_SIZE_D; // Assumes BLOCK_SIZE_D is multiple of WARP_SIZE
//     float acc_o[TILE_SIZE_D];

//     __nv_bfloat16* o_local_ptr = o_ptr + pid_q_j * stride_on;
//     for (int d_idx = 0; d_idx < TILE_SIZE_D; ++d_idx) {
//         // int d = tid * TILE_SIZE_D + d_idx; // Global 'd' index
//         if (d_idx < BLOCK_SIZE_D) {
//             acc_o[d_idx] = __bfloat162float(o_local_ptr[d_idx * stride_od]);
//         } else {
//             acc_o[d_idx] = 0.0f; // Should not happen if BLOCK_SIZE_D is multiple of WARP_SIZE
//         }
//     }

//     float lse = lse_ptr[pid_q_j * stride_lse_n];
//     const float m_ij_last = m_ij_last_ptr[pid_q_j];

//     // ===================================================================================
//     // SOFTWARE PIPELINING LOOP
//     // ===================================================================================
//     int current_stage = 0;
//     int next_stage = 1;

//     // Prefetch for the first iteration (block_id = 0)
//     // This part runs while the main loop would normally be doing computation
//     // but there's no prior computation to overlap with.
//     {
//         int t_pref = t_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + 0];
//         int real_token_index_pref = real_token_index_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + 0];

//         // Pointers for current prefetch stage
//         float* m_ij_sh_curr_pref = (current_stage == 0) ? m_ij_sh_stage0 : m_ij_sh_stage1;
//         float* l_ij_sh_curr_pref = (current_stage == 0) ? l_ij_sh_stage0 : l_ij_sh_stage1;
//         __nv_bfloat16* o_tiles_sh_curr_pref = (current_stage == 0) ? o_tiles_sh_stage0 : o_tiles_sh_stage1;
//         float* acc_o_scales_sh_curr_pref = (current_stage == 0) ? acc_o_scales_sh_stage0 : acc_o_scales_sh_stage1;

//         if (t_pref != -1) {
//             // Determine global pointers based on t_pref and real_token_index_pref
//             int real_block_pos_pref;
//             const float* l_ij_ptr_pref;
//             const __nv_bfloat16* o_tiles_ptr_pref;
//             const float* acc_o_scales_ptr_pref;
//             int64_t stride_l_ij_b_pref, stride_l_ij_n_pref;
//             int64_t stride_otb_pref, stride_otn_pref;
//             int64_t stride_acc_b_pref, stride_acc_n_pref;

//             if (t_pref == 0) {
//                 real_block_pos_pref = 0;
//                 l_ij_ptr_pref = l_ij_first_ptr;
//                 o_tiles_ptr_pref = o_tiles_first_ptr;
//                 acc_o_scales_ptr_pref = acc_o_scales_first_ptr;
//                 stride_l_ij_b_pref = stride_l_ij_fb; stride_l_ij_n_pref = stride_l_ij_fn;
//                 stride_otb_pref = stride_otfb; stride_otn_pref = stride_otfn;
//                 stride_acc_b_pref = stride_acc_fb; stride_acc_n_pref = stride_acc_fn;
//             } else {
//                 real_block_pos_pref = t_pref - 1;
//                 l_ij_ptr_pref = l_ij_rest_ptr;
//                 o_tiles_ptr_pref = o_tiles_rest_ptr;
//                 acc_o_scales_ptr_pref = acc_o_scales_rest_ptr;
//                 stride_l_ij_b_pref = stride_l_ij_rb; stride_l_ij_n_pref = stride_l_ij_rn;
//                 stride_otb_pref = stride_otrb; stride_otn_pref = stride_otrn;
//                 stride_acc_b_pref = stride_acc_rb; stride_acc_n_pref = stride_acc_rn;
//             }

//             // Load scalars for this thread and store to shared memory
//             m_ij_sh_curr_pref[tid] = m_ij_ptr[t_pref * stride_m_ij_b + pid_q_j * stride_m_ij_n];
//             l_ij_sh_curr_pref[tid] = l_ij_ptr_pref[real_block_pos_pref * stride_l_ij_b_pref + real_token_index_pref * stride_l_ij_n_pref];

//             // Cooperative load for o_tiles and acc_o_scales
//             const __nv_bfloat16* o_tiles_local_ptr_pref = o_tiles_ptr_pref + real_block_pos_pref * stride_otb_pref + real_token_index_pref * stride_otn_pref;
//             const float acc_o_scale_tile_pref = acc_o_scales_ptr_pref[real_block_pos_pref * stride_acc_b_pref + real_token_index_pref * stride_acc_n_pref];

//             for (int d_idx = 0; d_idx < TILE_SIZE_D; ++d_idx) {
//                 // int d = tid * TILE_SIZE_D + d_idx;
//                 o_tiles_sh_curr_pref[d_idx] = o_tiles_local_ptr_pref[d_idx * stride_otfd];
//             }
//             acc_o_scales_sh_curr_pref[tid] = acc_o_scale_tile_pref;
//         } else {
//             // If t_pref is -1, fill shared memory with safe defaults
//             m_ij_sh_curr_pref[tid] = 0.0f;
//             l_ij_sh_curr_pref[tid] = 0.0f;
//             acc_o_scales_sh_curr_pref[tid] = 0.0f;
//             for (int d_idx = 0; d_idx < TILE_SIZE_D; ++d_idx) {
//                 // int d = tid * TILE_SIZE_D + d_idx;
//                 o_tiles_sh_curr_pref[d_idx] = __float2bfloat16(0.0f);
//             }
//         }
//     } // End of initial prefetch block

//     // Main pipelined loop
//     for (int block_id = 0; block_id < TOPK; ++block_id) {
//         __syncthreads(); // Sync after prefetch/before compute

//         // Pointers for current compute stage (using data that was prefetched)
//         float* m_ij_sh_curr_comp = (current_stage == 0) ? m_ij_sh_stage0 : m_ij_sh_stage1;
//         float* l_ij_sh_curr_comp = (current_stage == 0) ? l_ij_sh_stage0 : l_ij_sh_stage1;
//         __nv_bfloat16* o_tiles_sh_curr_comp = (current_stage == 0) ? o_tiles_sh_stage0 : o_tiles_sh_stage1;
//         float* acc_o_scales_sh_curr_comp = (current_stage == 0) ? acc_o_scales_sh_stage0 : acc_o_scales_sh_stage1;

//         // Get 't' from shared memory for this block_id
//         const int t = t_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + block_id];

//         if (t != -1) {
//             // Get data for current compute iteration from shared memory
//             const float m_ij = m_ij_sh_curr_comp[tid];
//             const float l_ij = l_ij_sh_curr_comp[tid];
//             const float acc_o_scale_tile = acc_o_scales_sh_curr_comp[tid];

//             // --- LSE Update ---
//             const float delta = lse - m_ij;
//             const float log_delta = exp2_fast(delta) + l_ij;
//             lse = m_ij + log2_fast(log_delta);

//             // --- 'o' update from shared memory ---
//             for (int d_idx = 0; d_idx < TILE_SIZE_D; ++d_idx) {
//                 // int d = tid * TILE_SIZE_D + d_idx;
//                 float o_tile_val = __bfloat162float(o_tiles_sh_curr_comp[d_idx]);
//                 acc_o[d_idx] = o_tile_val + acc_o[d_idx] * acc_o_scale_tile;
//             }
//         }

//         // Prefetch data for the NEXT iteration (block_id + 1)
//         if (block_id + 1 < TOPK) {
//             int t_next = t_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + block_id + 1];
//             int real_token_index_next = real_token_index_shared[tid * (TOPK + SMEM_PAD_INT_OR_FLOAT) + block_id + 1];

//             // Pointers for next prefetch stage
//             float* m_ij_sh_next_pref = (next_stage == 0) ? m_ij_sh_stage0 : m_ij_sh_stage1;
//             float* l_ij_sh_next_pref = (next_stage == 0) ? l_ij_sh_stage0 : l_ij_sh_stage1;
//             __nv_bfloat16* o_tiles_sh_next_pref = (next_stage == 0) ? o_tiles_sh_stage0 : o_tiles_sh_stage1;
//             float* acc_o_scales_sh_next_pref = (next_stage == 0) ? acc_o_scales_sh_stage0 : acc_o_scales_sh_stage1;

//             if (t_next != -1) {
//                 // Determine global pointers based on t_next and real_token_index_next
//                 int real_block_pos_next;
//                 const float* l_ij_ptr_next;
//                 const __nv_bfloat16* o_tiles_ptr_next;
//                 const float* acc_o_scales_ptr_next;
//                 int64_t stride_l_ij_b_next, stride_l_ij_n_next;
//                 int64_t stride_otb_next, stride_otn_next;
//                 int64_t stride_acc_b_next, stride_acc_n_next;

//                 if (t_next == 0) {
//                     real_block_pos_next = 0;
//                     l_ij_ptr_next = l_ij_first_ptr;
//                     o_tiles_ptr_next = o_tiles_first_ptr;
//                     acc_o_scales_ptr_next = acc_o_scales_first_ptr;
//                     stride_l_ij_b_next = stride_l_ij_fb; stride_l_ij_n_next = stride_l_ij_fn;
//                     stride_otb_next = stride_otfb; stride_otn_next = stride_otfn;
//                     stride_acc_b_next = stride_acc_fb; stride_acc_n_next = stride_acc_fn;
//                 } else {
//                     real_block_pos_next = t_next - 1;
//                     l_ij_ptr_next = l_ij_rest_ptr;
//                     o_tiles_ptr_next = o_tiles_rest_ptr;
//                     acc_o_scales_ptr_next = acc_o_scales_rest_ptr;
//                     stride_l_ij_b_next = stride_l_ij_rb; stride_l_ij_n_next = stride_l_ij_rn;
//                     stride_otb_next = stride_otrb; stride_otn_next = stride_otrn;
//                     stride_acc_b_next = stride_acc_rb; stride_acc_n_next = stride_acc_rn;
//                 }

//                 // Load scalars for this thread and store to shared memory
//                 m_ij_sh_next_pref[tid] = m_ij_ptr[t_next * stride_m_ij_b + pid_q_j * stride_m_ij_n];
//                 l_ij_sh_next_pref[tid] = l_ij_ptr_next[real_block_pos_next * stride_l_ij_b_next + real_token_index_next * stride_l_ij_n_next];

//                 // Cooperative load for o_tiles and acc_o_scales
//                 const __nv_bfloat16* o_tiles_local_ptr_next = o_tiles_ptr_next + real_block_pos_next * stride_otb_next + real_token_index_next * stride_otn_next;
//                 const float acc_o_scale_tile_next = acc_o_scales_ptr_next[real_block_pos_next * stride_acc_b_next + real_token_index_next * stride_acc_n_next];

//                 for (int d_idx = 0; d_idx < TILE_SIZE_D; ++d_idx) {
//                     // int d = tid * TILE_SIZE_D + d_idx;
//                     o_tiles_sh_next_pref[d_idx] = o_tiles_local_ptr_next[d_idx * stride_otfd];
//                 }
//                 acc_o_scales_sh_next_pref[tid] = acc_o_scale_tile_next;
//             } else {
//                 // If t_next is -1, fill shared memory with safe defaults
//                 m_ij_sh_next_pref[tid] = 0.0f;
//                 l_ij_sh_next_pref[tid] = 0.0f;
//                 acc_o_scales_sh_next_pref[tid] = 0.0f;
//                 for (int d_idx = 0; d_idx < TILE_SIZE_D; ++d_idx) {
//                     // int d = tid * TILE_SIZE_D + d_idx;
//                     o_tiles_sh_next_pref[d_idx] = __float2bfloat16(0.0f);
//                 }
//             }
//         }

//         // Swap stages for the next iteration
//         current_stage = 1 - current_stage;
//         next_stage = 1 - next_stage;
//     }
//     __syncthreads(); // Ensure last iteration's computation is complete

//     // ===================================================================================
//     // FINALIZATION: Apply final scale and store results
//     // ===================================================================================
//     const float final_scale = exp2_fast(m_ij_last - lse);

//     for (int d_idx = 0; d_idx < TILE_SIZE_D; ++d_idx) {
//         //  int d = tid * TILE_SIZE_D + d_idx;
//          if (d_idx < BLOCK_SIZE_D) { // Check only needed if BLOCK_SIZE_D not multiple of WARP_SIZE

//             acc_o[d_idx] *= final_scale;
//             o_ptr[pid_q_j * stride_on + d_idx * stride_od] = __float2bfloat16_rn(acc_o[d_idx]);
//         }
//     }

//     lse_ptr[pid_q_j * stride_lse_n] = lse;
// }


// Naive correct
// template<int BLOCK_SIZE_D, int TOPK>
// __global__ void reduce_kernel_cuda(
//     // Pointers
//     float* lse_ptr,
//     const float* m_ij_ptr,
//     const float* l_ij_first_ptr,
//     const float* l_ij_rest_ptr,
//     const float* m_ij_last_ptr,
//     __nv_bfloat16* o_ptr,                  // <-- CORRECTED TYPE
//     const __nv_bfloat16* o_tiles_first_ptr,  // <-- CORRECTED TYPE
//     const __nv_bfloat16* o_tiles_rest_ptr,   // <-- CORRECTED TYPE
//     const float* acc_o_scales_first_ptr,
//     const float* acc_o_scales_rest_ptr,
//     const int* t_ptr,
//     const int* token_index_mapping_ptr,
//     // Scalars
//     int start_head_id,
//     int total_len,
//     // Strides
//     int64_t stride_lse_n,
//     int64_t stride_m_ij_b, int64_t stride_m_ij_n,
//     int64_t stride_l_ij_fb, int64_t stride_l_ij_fn,
//     int64_t stride_l_ij_rb, int64_t stride_l_ij_rn,
//     int64_t stride_on, int64_t stride_od,
//     int64_t stride_otfb, int64_t stride_otfn, int64_t stride_otfd,
//     int64_t stride_otrb, int64_t stride_otrn, int64_t stride_otrd,
//     int64_t stride_acc_fb, int64_t stride_acc_fn,
//     int64_t stride_acc_rb, int64_t stride_acc_rn,
//     int64_t stride_tn, int64_t stride_tk,
//     int64_t stride_tim_b, int64_t stride_tim_n
// ) {
//     // ===================================================================================
//     // KERNEL SETUP: Map threads to tokens
//     // ===================================================================================
//     const int tid = threadIdx.x;
//     const int pid_q_j = blockIdx.x * blockDim.x + tid;

//     if (pid_q_j >= total_len) {
//         return;
//     }

//     // ===================================================================================
//     // OPTIMIZATION: Bulk load 't' indices into shared memory
//     // ===================================================================================
//     extern __shared__ int t_shared_storage[]; // Dynamically sized shared memory
//     int* t_shared = t_shared_storage; // [BLOCK_DIM_X][TOPK]

//     const int* t_global_ptr = t_ptr + pid_q_j * stride_tn;

//     // Each thread loads its own TOPK indices into its row in shared memory.
//     // This can be further optimized for coalescing, but is a correct starting point.
//     for (int k = 0; k < TOPK; ++k) {
//         t_shared[tid * TOPK + k] = t_global_ptr[k * stride_tk];
        
//     }
//     __syncthreads(); // Ensure all 't' indices are loaded before proceeding.

//     // ===================================================================================
//     // INITIALIZATION: Load initial 'o' and 'lse' state
//     // ===================================================================================
//     // Accumulator for 'o' must be in high precision (float32)
//     float acc_o[BLOCK_SIZE_D];

//     // Load initial 'o' values, converting from bf16 to float32
//     __nv_bfloat16* o_local_ptr = o_ptr + pid_q_j * stride_on;
//     for (int d = 0; d < BLOCK_SIZE_D; ++d) {
//         if (d * stride_od < total_len * stride_on) { // Boundary check
//             acc_o[d] = __bfloat162float(o_local_ptr[d * stride_od]);
//         } else {
//             acc_o[d] = 0.0f;
//         }
//     }

//     float lse = lse_ptr[pid_q_j * stride_lse_n];
//     const float m_ij_last = m_ij_last_ptr[pid_q_j];

//     // ===================================================================================
//     // MAIN REDUCTION LOOP
//     // ===================================================================================
//     for (int block_id = 0; block_id < TOPK; ++block_id) {
//         const int t = t_shared[tid * TOPK + block_id];

//         if (t != -1) {
//             // Triton's branching logic translated to CUDA
//             int real_block_pos;
//             const float* l_ij_ptr;
//             const __nv_bfloat16* o_tiles_ptr;
//             const float* acc_o_scales_ptr;
//             int64_t stride_l_ij_b, stride_l_ij_n;
//             int64_t stride_otb, stride_otn;
//             int64_t stride_acc_b, stride_acc_n;

//             if (t == 0) {
//                 real_block_pos = 0;
//                 l_ij_ptr = l_ij_first_ptr;
//                 o_tiles_ptr = o_tiles_first_ptr;
//                 acc_o_scales_ptr = acc_o_scales_first_ptr;
//                 stride_l_ij_b = stride_l_ij_fb; stride_l_ij_n = stride_l_ij_fn;
//                 stride_otb = stride_otfb; stride_otn = stride_otfn;
//                 stride_acc_b = stride_acc_fb; stride_acc_n = stride_acc_fn;
//             } else {
//                 real_block_pos = t - 1;
//                 l_ij_ptr = l_ij_rest_ptr;
//                 o_tiles_ptr = o_tiles_rest_ptr;
//                 acc_o_scales_ptr = acc_o_scales_rest_ptr;
//                 stride_l_ij_b = stride_l_ij_rb; stride_l_ij_n = stride_l_ij_rn;
//                 stride_otb = stride_otrb; stride_otn = stride_otrn;
//                 stride_acc_b = stride_acc_rb; stride_acc_n = stride_acc_rn;
//             }

//             const int real_token_index = token_index_mapping_ptr[t * stride_tim_b + pid_q_j * stride_tim_n];

//             // --- Scalar loads from global memory (potential future optimization) ---
//             const float m_ij = m_ij_ptr[t * stride_m_ij_b + pid_q_j * stride_m_ij_n];
//             const float l_ij = l_ij_ptr[real_block_pos * stride_l_ij_b + real_token_index * stride_l_ij_n];

//             // --- LSE Update ---
//             const float delta = lse - m_ij;
//             const float log_delta = exp2_fast(delta) + l_ij;
//             lse = m_ij + log2_fast(log_delta);

//             // --- Vector loads and 'o' update ---
//             const __nv_bfloat16* o_tiles_local_ptr = o_tiles_ptr + real_block_pos * stride_otb + real_token_index * stride_otn;
//             const float acc_o_scale_tile = acc_o_scales_ptr[real_block_pos * stride_acc_b + real_token_index * stride_acc_n];

//             // This loop loads o_tiles (bf16), converts to float, and updates acc_o (float)
//             for (int d = 0; d < BLOCK_SIZE_D; ++d) {
//                 float o_tile_val = __bfloat162float(o_tiles_local_ptr[d * stride_otfd]);
//                 acc_o[d] = o_tile_val + acc_o[d] * acc_o_scale_tile;
//             }
//         }
//     }

//     // ===================================================================================
//     // FINALIZATION: Apply final scale and store results
//     // ===================================================================================
//     const float final_scale = exp2_fast(m_ij_last - lse);

//     // Store final 'o', converting from float32 accumulator to bf16 storage
//     for (int d = 0; d < BLOCK_SIZE_D; ++d) {
//          if (d * stride_od < total_len * stride_on) { // Boundary check
//             acc_o[d] *= final_scale;
//             o_local_ptr[d * stride_od] = __float2bfloat16_rn(acc_o[d]);
//         }
//     }

//     // Store final 'lse'
//     lse_ptr[pid_q_j * stride_lse_n] = lse;
// }


// Host function to launch the CUDA kernel
void lse_reduce_kernel_launcher(
    torch::Tensor lse,
    torch::Tensor m_ij,
    torch::Tensor l_ij_first,
    torch::Tensor l_ij_rest,
    torch::Tensor m_ij_last,
    torch::Tensor o,
    torch::Tensor o_tiles_first,
    torch::Tensor o_tiles_rest,
    torch::Tensor acc_o_scales_first,
    torch::Tensor acc_o_scales_rest,
    torch::Tensor t,
    torch::Tensor token_index_mapping,
    int start_head_id,
    int total_len,
    int topk
) {
    // Ensure all tensors are on the same CUDA device
    const auto device = t.device();
    TORCH_CHECK(lse.device() == device, "All tensors must be on the same device");
    TORCH_CHECK(o.device() == device, "All tensors must be on the same device");

    // Validate data types
    TORCH_CHECK(o.scalar_type() == torch::kBFloat16, "o must be bfloat16");
    TORCH_CHECK(o_tiles_first.scalar_type() == torch::kBFloat16, "o_tiles_first must be bfloat16");
    TORCH_CHECK(o_tiles_rest.scalar_type() == torch::kBFloat16, "o_tiles_rest must be bfloat16");
    TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be float32");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, "t must be int32");

    // Get problem dimensions
    const int num_heads = t.size(0);
    const int grid_z = num_heads; // One grid dimension for heads is common, but your triton code doesn't use it.
                                 // Let's stick to the Triton launch grid for now.

    const int num_qz_loop = total_len; // This seems to be the full length
    const dim3 grid( (total_len + WARP_SIZE - 1) / WARP_SIZE, 1, 1);
    const dim3 block(WARP_SIZE, 1, 1);

    // In your host-side code (before launching the kernel):

    // BLOCK_SIZE_D and TOPK are template parameters, so they'll be known at compile time
    // or passed as arguments to a wrapper function.
    // Let's assume BLOCK_SIZE_D and TOPK are available here.


    const int BLOCK_SIZE_D = o.size(2); // Assuming o is N x H x D
    TORCH_CHECK(BLOCK_SIZE_D == 64 || BLOCK_SIZE_D == 128, "Unsupported BLOCK_SIZE_D");


    // Calculate the offsets exactly as in the kernel
    size_t offset_t_shared = 0;
    size_t offset_real_token_index_shared = offset_t_shared + sizeof(int) * WARP_SIZE * (topk + SMEM_PAD_INT_OR_FLOAT);

    size_t offset_m_ij_stage0 = offset_real_token_index_shared + sizeof(int) * WARP_SIZE * (topk + SMEM_PAD_INT_OR_FLOAT);
    size_t offset_m_ij_stage1 = offset_m_ij_stage0 + sizeof(float) * WARP_SIZE;

    size_t offset_l_ij_stage0 = offset_m_ij_stage1 + sizeof(float) * WARP_SIZE;
    size_t offset_l_ij_stage1 = offset_l_ij_stage0 + sizeof(float) * WARP_SIZE;

    // The total shared memory size is the offset of the *last element* + its size.
    // The last array starts at offset_acc_o_scales_stage1 and contains WARP_SIZE floats.
    size_t shared_mem_size = offset_l_ij_stage1 + sizeof(float) * WARP_SIZE;
    // --- BEGIN DEBUG PRINT ---
    // printf("[Host] Launching reduce_kernel_cuda with configuration:\n");
    // printf("       - total_len: %d, TOPK: %d, BLOCK_SIZE_D: %d\n", total_len, topk, BLOCK_SIZE_D);
    // printf("       - Grid: (%u, %u, %u), Block: (%u, %u, %u)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    // printf("       - Shared Memory per Block: %zu bytes\n", shared_mem_size);
    // --- END DEBUG PRINT ---
    // This is a simplified dispatch. You might need a more complex one if D and TOPK vary a lot.
    if (BLOCK_SIZE_D == 128) {
        if (topk == 16) {
             lse_reduce_kernel_cuda<128, 16><<<grid, block, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
                lse.data_ptr<float>(), m_ij.data_ptr<float>(), l_ij_first.data_ptr<float>(), l_ij_rest.data_ptr<float>(),
                m_ij_last.data_ptr<float>(),
                t.data_ptr<int>(), token_index_mapping.data_ptr<int>(),
                start_head_id, total_len,
                lse.stride(1),
                m_ij.stride(1), m_ij.stride(2),
                l_ij_first.stride(1), l_ij_first.stride(2),
                l_ij_rest.stride(1), l_ij_rest.stride(2),
                t.stride(1), t.stride(2),
                token_index_mapping.stride(1), token_index_mapping.stride(2)
            );
        } // Add else-if for other TOPK values
    } // Add else-if for other BLOCK_SIZE_D values

    // Check for any errors during kernel launch
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


// // Host function to launch the CUDA kernel
// void o_reduce_kernel_launcher(
//     torch::Tensor lse,
//     torch::Tensor m_ij,
//     torch::Tensor l_ij_first,
//     torch::Tensor l_ij_rest,
//     torch::Tensor m_ij_last,
//     torch::Tensor o,
//     torch::Tensor o_tiles_first,
//     torch::Tensor o_tiles_rest,
//     torch::Tensor acc_o_scales_first,
//     torch::Tensor acc_o_scales_rest,
//     torch::Tensor t,
//     torch::Tensor token_index_mapping,
//     int start_head_id,
//     int total_len,
//     int topk
// ) {
//     // Ensure all tensors are on the same CUDA device
//     const auto device = t.device();
//     TORCH_CHECK(lse.device() == device, "All tensors must be on the same device");
//     TORCH_CHECK(o.device() == device, "All tensors must be on the same device");

//     // Validate data types
//     TORCH_CHECK(o.scalar_type() == torch::kBFloat16, "o must be bfloat16");
//     TORCH_CHECK(o_tiles_first.scalar_type() == torch::kBFloat16, "o_tiles_first must be bfloat16");
//     TORCH_CHECK(o_tiles_rest.scalar_type() == torch::kBFloat16, "o_tiles_rest must be bfloat16");
//     TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be float32");
//     TORCH_CHECK(t.scalar_type() == torch::kInt32, "t must be int32");

//     // Get problem dimensions
//     const int num_heads = t.size(0);
//     const int grid_z = num_heads; // One grid dimension for heads is common, but your triton code doesn't use it.
//                                  // Let's stick to the Triton launch grid for now.

//     const int num_qz_loop = total_len; // This seems to be the full length
//     const dim3 grid( (total_len + WARP_SIZE - 1) / WARP_SIZE, 1, 1);
//     const dim3 block(WARP_SIZE, 1, 1);

//     // In your host-side code (before launching the kernel):

//     // BLOCK_SIZE_D and TOPK are template parameters, so they'll be known at compile time
//     // or passed as arguments to a wrapper function.
//     // Let's assume BLOCK_SIZE_D and TOPK are available here.


//     const int BLOCK_SIZE_D = o.size(2); // Assuming o is N x H x D
//     TORCH_CHECK(BLOCK_SIZE_D == 64 || BLOCK_SIZE_D == 128, "Unsupported BLOCK_SIZE_D");


//     // Calculate the offsets exactly as in the kernel
//     size_t offset_t_shared = 0;
//     size_t offset_real_token_index_shared = offset_t_shared + sizeof(int) * WARP_SIZE * (topk + SMEM_PAD_INT_OR_FLOAT);

//     size_t offset_m_ij_stage0 = offset_real_token_index_shared + sizeof(int) * WARP_SIZE * (topk + SMEM_PAD_INT_OR_FLOAT);
//     size_t offset_m_ij_stage1 = offset_m_ij_stage0 + sizeof(float) * WARP_SIZE;

//     size_t offset_l_ij_stage0 = offset_m_ij_stage1 + sizeof(float) * WARP_SIZE;
//     size_t offset_l_ij_stage1 = offset_l_ij_stage0 + sizeof(float) * WARP_SIZE;

//     // The total shared memory size is the offset of the *last element* + its size.
//     // The last array starts at offset_acc_o_scales_stage1 and contains WARP_SIZE floats.
//     size_t shared_mem_size = offset_l_ij_stage1 + sizeof(float) * WARP_SIZE;
//     // --- BEGIN DEBUG PRINT ---
//     // printf("[Host] Launching reduce_kernel_cuda with configuration:\n");
//     // printf("       - total_len: %d, TOPK: %d, BLOCK_SIZE_D: %d\n", total_len, topk, BLOCK_SIZE_D);
//     // printf("       - Grid: (%u, %u, %u), Block: (%u, %u, %u)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
//     // printf("       - Shared Memory per Block: %zu bytes\n", shared_mem_size);
//     // --- END DEBUG PRINT ---
//     // This is a simplified dispatch. You might need a more complex one if D and TOPK vary a lot.
//     if (BLOCK_SIZE_D == 128) {
//         if (topk == 16) {
//              o_reduce_kernel_cuda<128, 16><<<grid, block, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
//                 lse.data_ptr<float>(),
//                 m_ij_last.data_ptr<float>(),
//                 reinterpret_cast<__nv_bfloat16*>(o.data_ptr()),
//                 reinterpret_cast<const __nv_bfloat16*>(o_tiles_first.data_ptr()),
//                 reinterpret_cast<const __nv_bfloat16*>(o_tiles_rest.data_ptr()),
//                 acc_o_scales_first.data_ptr<float>(), acc_o_scales_rest.data_ptr<float>(),
//                 t.data_ptr<int>(), token_index_mapping.data_ptr<int>(),
//                 start_head_id, total_len,
//                 lse.stride(1),
//                 o.stride(0), o.stride(2),
//                 o_tiles_first.stride(1), o_tiles_first.stride(2), o_tiles_first.stride(3),
//                 o_tiles_rest.stride(1), o_tiles_rest.stride(2), o_tiles_rest.stride(3),
//                 acc_o_scales_first.stride(1), acc_o_scales_first.stride(2),
//                 acc_o_scales_rest.stride(1), acc_o_scales_rest.stride(2),
//                 t.stride(1), t.stride(2),
//                 token_index_mapping.stride(1), token_index_mapping.stride(2)
//             );
//         } // Add else-if for other TOPK values
//     } // Add else-if for other BLOCK_SIZE_D values

//     // Check for any errors during kernel launch
//     C10_CUDA_KERNEL_LAUNCH_CHECK();
// }

// Host function to launch the CUDA kernel
void o_reduce_kernel_launcher(
    torch::Tensor lse,
    torch::Tensor m_ij,
    torch::Tensor l_ij_first,
    torch::Tensor l_ij_rest,
    torch::Tensor m_ij_last,
    torch::Tensor o,
    torch::Tensor o_tiles_first,
    torch::Tensor o_tiles_rest,
    torch::Tensor acc_o_scales_first,
    torch::Tensor acc_o_scales_rest,
    torch::Tensor t,
    torch::Tensor token_index_mapping,
    int start_head_id,
    int total_len,
    int topk
) {
    // Ensure all tensors are on the same CUDA device
    const auto device = t.device();
    TORCH_CHECK(lse.device() == device, "All tensors must be on the same device");
    TORCH_CHECK(o.device() == device, "All tensors must be on the same device");

    // Validate data types
    TORCH_CHECK(o.scalar_type() == torch::kBFloat16, "o must be bfloat16");
    TORCH_CHECK(o_tiles_first.scalar_type() == torch::kBFloat16, "o_tiles_first must be bfloat16");
    TORCH_CHECK(o_tiles_rest.scalar_type() == torch::kBFloat16, "o_tiles_rest must be bfloat16");
    TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be float32");
    TORCH_CHECK(t.scalar_type() == torch::kInt32, "t must be int32");

    // Fixed template parameters
    constexpr int BLOCK_SIZE_D = 128;
    constexpr int TOPK = 16;
    constexpr int THREADS_PER_TOKEN = 32;
    constexpr int TOKENS_PER_BLOCK = 4;
    constexpr int THREADS_PER_BLOCK = 128;

    // Validate topk
    TORCH_CHECK(topk == TOPK, "topk must be ", TOPK);

    // Grid and block configuration
    const int num_blocks = (total_len + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
    const dim3 grid(num_blocks, 1, 1);
    const dim3 block(THREADS_PER_BLOCK, 1, 1);

    // Calculate shared memory size based on kernel implementation
    size_t shared_mem_size = 0;
    
    // Static data for all tokens
    shared_mem_size += sizeof(int) * TOKENS_PER_BLOCK * TOPK; // t_shared
    shared_mem_size += sizeof(int) * TOKENS_PER_BLOCK * TOPK; // real_token_index_shared
    
    // Double buffered data - Stage 0
    shared_mem_size += sizeof(float) * TOKENS_PER_BLOCK; // acc_o_scales_stage0
    shared_mem_size += sizeof(float) * TOKENS_PER_BLOCK * BLOCK_SIZE_D; // o_tiles_stage0 (as float)
    
    // Double buffered data - Stage 1
    shared_mem_size += sizeof(float) * TOKENS_PER_BLOCK; // acc_o_scales_stage1
    shared_mem_size += sizeof(float) * TOKENS_PER_BLOCK * BLOCK_SIZE_D; // o_tiles_stage1 (as float)
    
    // Additional shared memory for constants per token
    shared_mem_size += sizeof(float) * TOKENS_PER_BLOCK; // shared_lse
    shared_mem_size += sizeof(float) * TOKENS_PER_BLOCK; // shared_m_ij_last
    shared_mem_size += sizeof(float) * TOKENS_PER_BLOCK; // shared_final_scale

    // Set CUDA device
    const at::cuda::CUDAGuard device_guard(device);

    // Launch kernel with fixed template parameters
    o_reduce_kernel_cuda<BLOCK_SIZE_D, TOPK><<<grid, block, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
        // Pointers
        lse.data_ptr<float>(),
        m_ij_last.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(o.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(o_tiles_first.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(o_tiles_rest.data_ptr<at::BFloat16>()),
        acc_o_scales_first.data_ptr<float>(),
        acc_o_scales_rest.data_ptr<float>(),
        t.data_ptr<int>(),
        token_index_mapping.data_ptr<int>(),
        // Scalars
        start_head_id,
        total_len,
        // Strides
        lse.stride(0),                    // stride_lse_n
        o.stride(0), o.stride(1),         // stride_on, stride_od
        o_tiles_first.stride(0), o_tiles_first.stride(1), o_tiles_first.stride(2), // stride_otfb, stride_otfn, stride_otfd
        o_tiles_rest.stride(0), o_tiles_rest.stride(1), o_tiles_rest.stride(2),    // stride_otrb, stride_otrn, stride_otrd
        acc_o_scales_first.stride(0), acc_o_scales_first.stride(1),                // stride_acc_fb, stride_acc_fn
        acc_o_scales_rest.stride(0), acc_o_scales_rest.stride(1),                  // stride_acc_rb, stride_acc_rn
        t.stride(0), t.stride(1),                                                  // stride_tn, stride_tk
        token_index_mapping.stride(0), token_index_mapping.stride(1)               // stride_tim_b, stride_tim_n
    );

    // Check for kernel launch errors
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    
    // Optional: Check for kernel execution errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel execution failed: ", cudaGetErrorString(err));
}
