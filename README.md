# CUDA Kernel for Matrix-Matrix Multiplication on Nvidia GPUs

This code accompanies the blog post [Matrix Multiplication Faster Than Nvidia, Sometimes](https://indii.org/blog/gpu-matrix-multiply). It provides a CUDA kernel for single-precision matrix-matrix multiplication, with two notable features:

* use of a Hilbert curve to improve L2 cache efficiency,
* avoidance of synchronization across whole thread blocks, instead replaced with synchronization across half and quarter blocks.


## License

This is open source software. It is licensed under the Apache License,
Version 2.0 (the "License"); you may not use it except in compliance with the
License. You may obtain a copy of the License at
<http://www.apache.org/licenses/LICENSE-2.0>.


## Requirements

You will need:

* an Nvidia graphics card,
* a working [CUDA](https://developer.nvidia.com/cuda-downloads) installation,
* [cmake](https://cmake.org) to build the code.

The code as been tested with an Nvidia GeForce RTX 4080 Laptop GPU, using CUDA 12.6.1 on a laptop running Ubuntu 24.04.


## Building

Build with:

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    cmake --build .


## Running

From within that same `build` directory, run with:

    ./gemm

A table of results is output in Markdown format. You may want to tweak the actual tests that are run by editing `src/gemm.cu` (at the bottom) before compiling, especially if you need to reduce the number of trials or remove the larger matrix sizes to fit within memory constraints. Without changes, a GPU with at least 4 GB of device memory is ideal.

The code uses 32-bit array indexing, and will not work with matrices larger than 32768x32768 without modification to avoid integer overflow.

> Consider sharing your results as a [discussion](https://github.com/lawmurray/gpu-gemm/discussions). You can copy and paste the output table directly, as it is in GitHub compatible Markdown.


## Benchmarking

For the purposes of benchmarking, refer to the blog post [Matrix Multiplication Faster than Nvidia, Sometimes](https://indii.org/blog/gpu-matrix-multiply) for a discussion of some appropriate protocols.

You may wish to lock the clock and memory speed on your GPU for benchmarking purpoes (or, you may not, refer to the blog post). To do so, run the following commands:

    sudo nvidia-smi --lock-gpu-clocks=1150
    sudo nvidia-smi --lock-memory-clocks=6000

Changing the numbers as desired. Once benchmarking is complete, unlock them again with:

    sudo nvidia-smi --reset-gpu-clocks
    sudo nvidia-smi --reset-memory-clocks


## Contributing

Contributions are welcome. This is prototype not production code, so a contributions should aim at improving understanding of this particular computation. That might include:

* Running the code on your own system and reporting the results. The output of the program is a Markdown table that you can easily copy and paste into a [discussion](https://github.com/lawmurray/gpu-gemm/discussions).
* Improving the performance of the code. Please send a [pull request](https://github.com/lawmurray/gpu-gemm/pulls), and perhaps consider writing a blog post or the like on the improvement.
* Improving the benchmarking protocol. Perhaps you think the methodology can be improved for more accurate measurement, or there is an interesting scenario that is not currently considered. Again, please send a [pull request](https://github.com/lawmurray/gpu-gemm/pulls), or [start a discussion](https://github.com/lawmurray/gpu-gemm/discussions) if required.
* Expanding to new use cases such as half precision or double precision.
* Expanding to new hardware.
* Fixing bugs if found.

Of course, these are just suggestions and not an exhaustive list.


## Contact

Lawrence Murray, <https://indii.org>, <lawrence@indii.org>.
