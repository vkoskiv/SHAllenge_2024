#include <stdio.h>
#include <openssl/sha.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

volatile int max_leading_zeroes = 0;

void generate_random_string(char *str, size_t length, unsigned int *seed) {
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/";
    size_t charset_size = sizeof(charset) - 1;

    for (size_t i = 0; i < length; ++i) {
        int key = rand() % charset_size;
        str[i] = charset[key];
    }
    str[length] = '\0';
}

void generate_random_string48(char *str, size_t length, struct drand48_data *rngbuf) {
    const char charset[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/";
    size_t charset_size = sizeof(charset) - 1;
    double rand_val;

    for (size_t i = 0; i < length; ++i) {
    	drand48_r(rngbuf, &rand_val);
    	int idx = (int)(rand_val * charset_size);
    	str[i] = charset[idx];
    }
    str[length] = '\0';
}

void compute_sha256(const char *str, size_t len, unsigned char output[SHA256_DIGEST_LENGTH]) {
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, str, len);
    SHA256_Final(output, &sha256);
}

void print_sha256(unsigned char hash[SHA256_DIGEST_LENGTH]) {
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
}

static inline int count_leading_zeroes_byte(uint8_t byte) {
    int count = 0;
    for (int i = 7; i >= 0; --i) {
        if ((byte & (1 << i)) == 0) {
            count++;
        } else {
            break;
        }
    }
    return count;
}

static inline int count_leading_zeroes(const uint8_t *data, size_t length) {
    int total_count = 0;

    for (size_t i = 0; i < length; ++i) {
        if (data[i] == 0) {
            total_count += 8;
        } else {
            total_count += count_leading_zeroes_byte(data[i]);
            break;
        }
    }

    return total_count;
}

void *thread_fn(void *arg) {
	struct drand48_data rngbuf;
	double result;
	unsigned int seed = *(unsigned int *)arg;
	char buf[128] = "vkoskiv/";
	char *nonce = &buf[8];

	long int thread_seed = seed ^ (pthread_self() << 16);
	srand48_r(thread_seed, &rngbuf);
	int local_max = 0;
    unsigned char hash[SHA256_DIGEST_LENGTH];
    uint64_t checked_hashes = 0;
	
	while (1) {
		generate_random_string48(nonce, 8, &rngbuf);
	    compute_sha256(buf, 16, hash);
	    int zeroes = count_leading_zeroes(hash, SHA256_DIGEST_LENGTH);
		checked_hashes++;

	    if (zeroes && zeroes > local_max) {
	    	local_max = zeroes;
	    }
	    if (local_max > max_leading_zeroes) {
	    	max_leading_zeroes = local_max;
	    	const time_t curTime = time(NULL);
			struct tm *time = localtime(&curTime);
			printf("[%d-%02d-%02d %02d:%02d:%02d] %s (%i,%lu): ",
				   time->tm_year + 1900,
				   time->tm_mon + 1,
				   time->tm_mday,
				   time->tm_hour,
				   time->tm_min,
				   time->tm_sec, buf, zeroes, checked_hashes);
		    print_sha256(hash);
		    // printf("%s (%i): ", buf, zeroes);
		    // print_sha256(hash);
	    } else {
	    	local_max = max_leading_zeroes;
	    }
	}
}

#define NTHREADS 8

int main() {
	char buf[128] = "vkoskiv/";
	char *nonce = &buf[8];

#ifdef STHREAD
	int seed = time(NULL);
	printf("Seed: %i\n", seed);
	srand(seed);
    unsigned char hash[SHA256_DIGEST_LENGTH];
    size_t max_zeroes = 0;
	while (1) {
		generate_random_string(nonce, 16, &seed);
	    compute_sha256(buf, 24, hash);
	    int zeroes = count_leading_zeroes(hash, SHA256_DIGEST_LENGTH);

	    if (zeroes && zeroes > max_zeroes) {
	    	max_zeroes = zeroes;
	    	const time_t curTime = time(NULL);
			struct tm *time = localtime(&curTime);
			printf("[%d-%02d-%02d %02d:%02d:%02d] %s (%i): ",
				   time->tm_year + 1900,
				   time->tm_mon + 1,
				   time->tm_mday,
				   time->tm_hour,
				   time->tm_min,
				   time->tm_sec, buf, zeroes);
		    print_sha256(hash);
	    }
	}

#else
	pthread_t threads[NTHREADS];
	int seed = time(NULL);
	printf("Seed: %i\n", seed);
	
	for (int i = 0; i < NTHREADS; ++i) {
		pthread_create(&threads[i], NULL, thread_fn, &seed);
	}

	for (int i = 0; i < NTHREADS; ++i) {
		pthread_join(threads[i], NULL);
	}
	
    // printf("SHA-256 hash of \"%s\":\n", input);
    // print_sha256(hash);
#endif

    return 0;
}
