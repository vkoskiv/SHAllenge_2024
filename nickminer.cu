#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
// #include <openssl/sha.h>
#include <curand_kernel.h>
#include "sha256.cuh"
#include <cuda.h>
#include <curl/curl.h>
#include <pthread.h>

# define SHA256_DIGEST_LENGTH    32

void *send_result(void *result) {
    CURL *curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "Failed to set up curl\n");
        return NULL;
    }
    const char *url = "https://shallenge.quirino.net/submit";
    char post_data[128];
    snprintf(post_data, 128, "submission=%s", (char *)result);
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_data);
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        fprintf(stderr, "failed to send req: %s\n", curl_easy_strerror(res));
    }
    curl_easy_cleanup(curl);
    return NULL;
}

// Just fire it off and ignore, to not block the GPU
void async_send_result(char *result) {
    pthread_attr_t attribs;
    pthread_t thread_id;
    pthread_attr_init(&attribs);
    pthread_attr_setdetachstate(&attribs, PTHREAD_CREATE_DETACHED);
    int ret = pthread_create(&thread_id, &attribs, send_result, (void *)result);
    pthread_setname_np(thread_id, "ResultSendTask");
    pthread_attr_destroy(&attribs);
    if (ret < 0) fprintf(stderr, "pthread oopsie\n");
}

__device__ void strcpy_device(char *dest, const char *src) {
    while ((*dest++ = *src++) != '\0');
}

// Charset for random generation
__device__ const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/";

// Device function to generate a random character from the charset
__device__ char random_char(curandState *state) {
    int index = curand(state) % (sizeof(charset) - 1);
    return charset[index];
}

// GPU Kernel to compute SHA-256 and count leading zeroes
__global__ void find_best_permutation(const char *prefix, char *best_permutation, int *max_zeroes, int prefix_length, int total_length, uint64_t seed, char *out_hash) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(seed, idx, 0, &state);
    char permutation[32];
    strcpy_device(permutation, prefix);
    // unsigned int local_seed = seed + idx;

    // Generate random characters for the rest of the string
    for (int i = prefix_length; i < total_length; i++) {
        permutation[i] = random_char(&state);
    }
    permutation[total_length] = '\0';

    // Compute SHA-256 hash
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (const BYTE *)permutation, total_length);
    sha256_final(&ctx, hash);

    // Count leading zeroes
    int leading_zeroes = 0;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        if (hash[i] == 0) {
            leading_zeroes += 8;
        } else {
            for (int j = 7; j >= 0; j--) {
                if ((hash[i] & (1 << j)) == 0) {
                    leading_zeroes++;
                } else {
                    break;
                }
                // if (hash[i] & (1 << j)) {
                //     i = SHA256_DIGEST_LENGTH;  // Break outer loop
                //     break;
                // }
                // leading_zeroes++;
            }
            break;
        }
    }

    // Update max_zeroes and best_permutation atomically
    if (leading_zeroes > *max_zeroes) {
        atomicMax(max_zeroes, leading_zeroes);
        // if (leading_zeroes == *max_zeroes) {
        //     strcpy_device(best_permutation, permutation);
        //     for (int i = 0; i < 32; ++i) {
        //         out_hash[i] = hash[i];
        //     }
        // }
        strcpy_device(best_permutation, permutation);
        for (int i = 0; i < 32; ++i) {
            out_hash[i] = hash[i];
        }
    }
}

void print_sha256(unsigned char hash[SHA256_DIGEST_LENGTH]) {
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

// Host code
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: <%s> <prefix> <total_len> [minimum_prefix_zeroes] [threads_per_block] [blocks]\n", argv[0]);
        return 1;
    }
    const char *prefix = argv[1];
    const int total_length = atoi(argv[2]);
    int h_max_zeroes = 0;
    if (argc > 3) {
        h_max_zeroes = atoi(argv[3]);
        printf("Using %i as initial size\n", h_max_zeroes);
    }

    // I have no idea how to choose good settings here
    // int threads_per_block = 256;
    // int number_of_blocks = 1024;
    int threads_per_block = 256;
    int number_of_blocks = 32768;

    if (argc > 4) {
        threads_per_block = atoi(argv[4]);
    }
    if (argc > 5) {
        number_of_blocks = atoi(argv[5]);
    }
    
    printf("Threads per block: %i, blocks: %i\n", threads_per_block, number_of_blocks);
    
    curl_global_init(CURL_GLOBAL_DEFAULT);
    int prefix_length = strlen(prefix);

    printf("Trying pattern %s", prefix);
    for (int i = prefix_length; i < total_length; ++i) {
        printf("*");
    }
    printf(" (%i)\n", total_length - prefix_length);
    
    char *d_prefix;
    char *d_best_permutation;
    int *d_max_zeroes;
    char *d_hash;

    cudaMalloc((void **)&d_prefix, prefix_length);
    cudaMalloc((void **)&d_best_permutation, total_length + 1);
    cudaMalloc((void **)&d_max_zeroes, sizeof(int));
    cudaMalloc((void **)&d_hash, SHA256_DIGEST_LENGTH);
    cudaMemset(d_best_permutation, 0, total_length + 1);
    cudaMemcpy(d_max_zeroes, &h_max_zeroes, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prefix, prefix, prefix_length, cudaMemcpyHostToDevice);
    cudaMemset(d_hash, 0, SHA256_DIGEST_LENGTH);

    checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
    char h_best_permutation[64] = { 0 };
    uint64_t seed = time(NULL);
    // unsigned int seed = 1718870720;
    printf("Starting seed is: %lu\n", seed);
    uint64_t tried_hashes = 0;
    const uint64_t theoretical_max = pow(strlen(charset), total_length - prefix_length);
    int perf_hashes_found = 0;
    while (true) {
        
        // checkCudaErrors((find_best_permutation<<<number_of_blocks, threads_per_block>>>(prefix, d_best_permutation, d_max_zeroes, prefix_length, total_length, seed)));
        find_best_permutation<<<number_of_blocks, threads_per_block>>>(d_prefix, d_best_permutation, d_max_zeroes, prefix_length, total_length, seed, d_hash);

        // Copy results back to host
        char temp_best_permutation[64] = { 0 };
        int temp_max_zeroes = 0;
        unsigned char h_hash[SHA256_DIGEST_LENGTH] = { 0 };
        cudaMemcpy(temp_best_permutation, d_best_permutation, total_length, cudaMemcpyDeviceToHost);
        cudaMemcpy(&temp_max_zeroes, d_max_zeroes, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_hash, d_hash, SHA256_DIGEST_LENGTH, cudaMemcpyDeviceToHost);

        // And check if we won the lottery
        if (temp_max_zeroes > h_max_zeroes) {
            h_max_zeroes = temp_max_zeroes;
            strcpy(h_best_permutation, temp_best_permutation);
            
            // Output the best permutation and number of leading zeroes
            const time_t curTime = time(NULL);
			struct tm *time = localtime(&curTime);
			printf("[%d-%02d-%02d %02d:%02d:%02d] %s (%i): ",
				   time->tm_year + 1900,
				   time->tm_mon + 1,
				   time->tm_mday,
				   time->tm_hour,
				   time->tm_min,
				   time->tm_sec, h_best_permutation, h_max_zeroes);
            print_sha256(h_hash);
            // Send result back to SHAllenge
            async_send_result(h_best_permutation);
            printf("%lu/%lu hashes\n", tried_hashes, theoretical_max);
        }
        tried_hashes += number_of_blocks * threads_per_block;
        // if (tried_hashes > 100000000) {
        //     printf("Perf measure, got %lu hashes\n", tried_hashes);
        //     break;
        // }
        seed += number_of_blocks * threads_per_block;
    }

    // Free device memory
    cudaFree(d_best_permutation);
    cudaFree(d_max_zeroes);
    cudaFree(d_prefix);
    curl_global_cleanup();
    return 0;
}
