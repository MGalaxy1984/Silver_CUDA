#include <iostream>
#include <stdio.h>
#include "AES.cuh"
#include <vector>
#include <windows.h>

#define NUM_THREADS 32

void cpu_print_block(uint8_t *b) {
    for (int i = 0; i < 16; i++) {
        printf("%02x", b[i]);
    }
}

__global__ void test(uint8_t *left_keys, uint8_t *right_keys) {
    printf("GPU Received Left Expanded Keys:\n");
    for (int i = 0; i < 11; i++) {
        print_block(&(left_keys[16 * i]));
        printf("\n");
    }
    printf("GPU Received Right Expanded Keys:\n");
    for (int i = 0; i < 11; i++) {
        print_block(&(right_keys[16 * i]));
        printf("\n");
    }
}

__global__ void test1(uint8_t *text, uint8_t *left_keys, uint8_t *right_keys) {
    uint8_t text_left[16];
    uint8_t text_right[16];
    aes_encryption(text, left_keys, text_left);
    aes_encryption(text, right_keys, text_right);
    printf("Text input = ");
    print_block(text);
    printf("\n");
    printf("left Key = ");
    print_block(&(left_keys[0]));
    printf("\n");
    printf("left output = ");
    print_block(text_left);
    printf("\n");
    printf("right Key = ");
    print_block(&(right_keys[0]));
    printf("\n");
    printf("right output = ");
    print_block(text_right);
    printf("\n");
}

__global__ void prng_generate(uint8_t *text, uint8_t *prng_keys) {
    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint8_t result[16];
    aes_encryption(&(text[16 * tid]), prng_keys, result);
    block_copy(result, &(text[16 * tid]));
}

__global__ void expand(uint8_t *text, uint8_t *left_keys, uint8_t *right_keys, int offset) {
    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ uint8_t text_left[NUM_THREADS * 16];
    __shared__ uint8_t text_right[NUM_THREADS * 16];
    uint8_t *text_in = &(text[16 * tid]);
    uint8_t *tmp_text_left = &(text_left[16 * threadIdx.x]);
    uint8_t *tmp_text_right = &(text_right[16 * threadIdx.x]);
    aes_encryption(text_in, left_keys, tmp_text_left);
    aes_encryption(text_in, right_keys, tmp_text_right);

    block_xor(tmp_text_left, text_in, tmp_text_left);
    block_xor(tmp_text_right, text_in, tmp_text_right);

//    if (tid == 1) {
//        printf("Text input = ");
//        print_block(text_in);
//        printf("\n");
////        printf("left Key = ");
////        print_block(&(left_keys[0]));
////        printf("\n");
//        printf("left output = ");
//        print_block(tmp_text_left);
//        printf("\n");
////        printf("right Key = ");
////        print_block(&(right_keys[0]));
////        printf("\n");
//        printf("right output = ");
//        print_block(tmp_text_right);
//        printf("\n");
//    }

    __syncthreads();
    uint32_t chunk_index = tid / 8;
    uint32_t left_addr = (chunk_index * 2) * 8 + (tid & 7);
    uint32_t right_addr = (chunk_index * 2 + 1) * 8 + (tid & 7);
    block_copy(tmp_text_left, &(text[16 * left_addr]));
    block_copy(tmp_text_right, &(text[16 * right_addr]));
}

__global__ void print_result(uint8_t *text, int block_num) {
    for (int i = 0; i < block_num; i++) {
        printf("%d: ", i);
        print_block(&(text[16 * i]));
        printf("\n");
    }
}

double single_run(int D) {

    int k = 1 << (D + 3);

    size_t key_size = 11 * 16 * sizeof(uint8_t);

//    printf("Left Expanded Keys:\n");

    uint8_t *cpu_left_key_space;
    cudaMallocHost((void **) &cpu_left_key_space, key_size);
    for (int i = 0; i < 16; i++) {
        cpu_left_key_space[i] = 0;
    }
    cpu_left_key_space[13] = 0x31;
    cpu_left_key_space[14] = 0x79;
    cpu_left_key_space[15] = 0x66;
    key_expansion(cpu_left_key_space);
//    for (int i = 0; i < 11; i++) {
//        cpu_print_block(&(cpu_left_key_space[16 * i]));
//        printf("\n");
//    }

//    printf("Right Expanded Keys:\n");

    uint8_t *cpu_right_key_space;
    cudaMallocHost((void **) &cpu_right_key_space, key_size);
    for (int i = 0; i < 16; i++) {
        cpu_right_key_space[i] = 0;
    }
    cpu_right_key_space[13] = 0x89;
    cpu_right_key_space[14] = 0x3c;
    cpu_right_key_space[15] = 0x39;
    key_expansion(cpu_right_key_space);
//    for (int i = 0; i < 11; i++) {
//        cpu_print_block(&(cpu_right_key_space[16 * i]));
//        printf("\n");
//    }

    uint8_t *cpu_prng_key_space;
    cudaMallocHost((void **) &cpu_prng_key_space, key_size);
    for (int i = 0; i < 16; i++) {
        cpu_prng_key_space[i] = 0;
    }
    key_expansion(cpu_prng_key_space);

    uint8_t *prng_keys;
    uint8_t *left_keys;
    uint8_t *right_keys;
    cudaMalloc((void **) &prng_keys, key_size);
    cudaMalloc((void **) &left_keys, key_size);
    cudaMalloc((void **) &right_keys, key_size);

    uint8_t *cpu_text;
    cudaMallocHost((void **) &cpu_text, k * 16 * sizeof(uint8_t));
    uint64_t nuance = 0xfefd1bb7e1d57b77;
    for (int i = 0; i < 8; i++) {
        memcpy(&(cpu_text[16 * i]), &nuance, 8);
        cpu_text[16 * i + 15] = i;
    }

    uint8_t *text;
    cudaMalloc((void **) &text, k * 16 * sizeof(uint8_t));

    cudaMemcpyAsync(prng_keys, cpu_prng_key_space, key_size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(left_keys, cpu_left_key_space, key_size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(right_keys, cpu_right_key_space, key_size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(text, cpu_text, 8 * 16 * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

//    test<<<1, 1>>>(left_keys, right_keys);

//    test1<<<1, 1>>>(text, left_keys, right_keys);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    prng_generate<<<1, 8>>>(text, prng_keys);
//    print_result<<<1, 1>>>(text, k);
    for (int i = 0; i < D; i++) {
        int num_thread = 8 << i;
        dim3 grid;
        dim3 block;
        if (num_thread <= NUM_THREADS) {
            grid.x = 1;
            block.x = num_thread;
        } else {
            grid.x = num_thread / NUM_THREADS;
            block.x = NUM_THREADS;
        }
        expand<<<grid, block, NUM_THREADS * 16>>>(text, left_keys, right_keys, num_thread);
    }
    cudaEventRecord(stop);

    cudaMemcpyAsync(cpu_text, text, k * 16 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
//    printf("Time = %f\n", milliseconds);

//    size_t free_mem;
//    size_t total_mem;
//    cudaMemGetInfo(&free_mem, &total_mem);
//    printf("Used Memory = %zu MB\n", (total_mem - free_mem)/1000000);
//    printf("Total Memory = %zu\n", total_mem);

//    print_result<<<1, 1>>>(text, k);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(cpu_left_key_space);
    cudaFreeHost(cpu_right_key_space);
    cudaFreeHost(cpu_prng_key_space);
    cudaFree(left_keys);
    cudaFree(right_keys);
    cudaFree(prng_keys);
    cudaFree(text);

    return milliseconds;
}

std::vector<double> run(int max_n, int test_times) {
    std::vector<double> runtime(max_n + 1);
    for (int n = 1; n <= max_n; n++) {
        double total_time = 0.0;
        for (int i = 0; i < test_times; i++) {
            auto single_run_time = single_run(n);
//            printf("D = %d, single runtime = %f\n", n, single_run_time);
            total_time += single_run_time;
        }
        total_time /= test_times;
        runtime[n] = total_time;
    }
    return runtime;
}

int main() {

    auto runtime_vector = run(20, 15);
    for (int i = 1; i < runtime_vector.size(); i++) {
        printf("D = %d, average runtime = %f\n", i, runtime_vector[i]);
    }
    for (int i = 1; i < runtime_vector.size(); i++) {
        printf("%f\n", runtime_vector[i]);
    }
    return 0;
}
