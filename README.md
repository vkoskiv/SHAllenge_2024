# My SHAllenge implementations

It's been a few months since [SHAllenge](https://shallenge.quirino.net/) happened, so I'm publishing the two programs I cobbled together to participate.

My initial implementation is the obvious, likely somewhat racy C99 + pthreads + OpenSSL implementation which briefly got me to around #70 in the top 100, before more people with GPUs joined in and dropped me back down. You'll find this implementation in `nickminer.c`. It just outputs hashes to stdout, along with a time stamp. Parameters are hard-coded in the .c file, and there is also a `STHREAD` define that picks the earlier single-threaded version.

I wanted to get a feel for CUDA programming, so to save some time reading documentation, I had a token predictor generate some code, where I could then fill in the gaps myself. You'll find my final implementation in `nickminer.cu`, and for comparison, the raw (broken) output of the token predictor is in `nickminer_template.cu`.

Later on I also linked in `libcurl(3)`, so it automatically submits new results to SHAllenge. This implementation is also slightly more polished, and allows you to specify options on the command line for prefix, length, a minimum difficulty (amount of binary zeroes), as well as the CUDA-specific parameters `threads_per_block` and `blocks`.

The code I got from the token predictor served as a nice brief on CUDA, but it isn't quite correct, and won't compile. It tries to use the OpenSSL SHA256 implementation in the CUDA kernel, which isn't possible as far as I'm aware, so I grabbed an [MIT licensed implementation](https://github.com/Horkyze/CudaSHA256/blob/master/sha256.cuh) which seems to work nicely.
It also assumes the C standard library `rand_r(3)` is available in CUDA, which is not the case. I swapped it out for `cuRAND`. I spent some more time fiddling with parameters and did some rudimentary benchmarking to find optimal values for `threads_per_block` and `blocks`.

I was running this CUDA program on my GTX 1070 card from June 2024 up until some point in July 2024 when the heat put out by my system started bothering me. I peaked at position #35, and I'm still in the top 100, at #62 at the time of writing.

# Build instructions

The C program depends on gcc and OpenSSL. The CUDA version depends on nvcc and libcurl.

To build the C99 program, run `make nickminer`

To build the CUDA version, run `make nickminer_cuda`

# Running

For the C99 program, you'll have to tweak the source code and recompile, and then just `./nickminer`

For the CUDA program, I've lost the benchmark data I used, but I found this invocation in my shell history, which should be pretty good:
`./nickminer_cuda "vkoskiv/" 20 44 32 4096`

SHAllenge instructions specify the following format for submissions:
~~~
Submit a string in the format "{username}/{nonce}", where:

    username: 1-32 characters from a-zA-Z0-9_-
    nonce: 1-64 characters from Base64 (a-zA-Z0-9+/)
~~~

# License

The C99 implementation is (c) me, and MIT-licensed. I'm not confident I can claim rights to code I got from a commercial token predictor run by a company that isn't very *open* about where they sourced their training data, but for the bits of `nickminer.cu` I wrote with my very own OI, the MIT license should be applied.

`sha256.cuh` comes from [https://github.com/Horkyze/CudaSHA256](https://github.com/Horkyze/CudaSHA256), and is MIT-licensed. The license file is available verbatim in LICENSE_CudaSHA256.
