#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <openssl/sha.h>
#include <cuda.h>

// Charset for random generation
const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/";

// Device function to generate a random character from the charset
__device__ char random_char(unsigned int *seed) {
    int index = rand_r(seed) % (sizeof(charset) - 1);
    return charset[index];
}

// GPU Kernel to compute SHA-256 and count leading zeroes
__global__ void find_best_permutation(const char *prefix, char *best_permutation, int *max_zeroes, int prefix_length, int total_length, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    char permutation[256];
    strcpy(permutation, prefix);
    unsigned int local_seed = seed + idx;

    // Generate random characters for the rest of the string
    for (int i = prefix_length; i < total_length; i++) {
        permutation[i] = random_char(&local_seed);
    }
    permutation[total_length] = '\0';

    // Compute SHA-256 hash
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, permutation, total_length);
    SHA256_Final(hash, &sha256);

    // Count leading zeroes
    int leading_zeroes = 0;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        for (int j = 7; j >= 0; j--) {
            if (hash[i] & (1 << j)) {
                i = SHA256_DIGEST_LENGTH;  // Break outer loop
                break;
            }
            leading_zeroes++;
        }
    }

    // Update max_zeroes and best_permutation atomically
    atomicMax(max_zeroes, leading_zeroes);
    if (leading_zeroes == *max_zeroes) {
        strcpy(best_permutation, permutation);
    }
}

// Host code
int main() {
    const char *prefix = "example";  // Prefix specified
    int prefix_length = strlen(prefix);
    int total_length = 16;  // Total length of the string to be generated

    char *d_best_permutation;
    int *d_max_zeroes;

    // Allocate memory on the device
    cudaMalloc((void **)&d_best_permutation, 256 * sizeof(char));
    cudaMalloc((void **)&d_max_zeroes, sizeof(int));
    cudaMemset(d_max_zeroes, 0, sizeof(int));

    // Launch kernel with enough threads to cover permutations
    int threads_per_block = 256;
    int number_of_blocks = 1024;  // Adjust based on the desired number of permutations
    unsigned int seed = time(NULL);
    find_best_permutation<<<number_of_blocks, threads_per_block>>>(prefix, d_best_permutation, d_max_zeroes, prefix_length, total_length, seed);

    // Copy results back to host
    char h_best_permutation[256];
    int h_max_zeroes;
    cudaMemcpy(h_best_permutation, d_best_permutation, 256 * sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max_zeroes, d_max_zeroes, sizeof(int), cudaMemcpyDeviceToHost);

    // Output the best permutation and number of leading zeroes
    printf("Best permutation: %s\n", h_best_permutation);
    printf("Number of leading zeroes: %d\n", h_max_zeroes);

    // Free device memory
    cudaFree(d_best_permutation);
    cudaFree(d_max_zeroes);

    return 0;
}

