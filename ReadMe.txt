This code has been tested on Ubuntu 14.01, minimum compiler requirements:

-GCC version 4.8.4
-NVIDIA CUDA version 7.5

Compile Using :
nvcc -arch=compute_35 -code=sm_35 -std=c++11 specFile.cpp spectrum.cpp G-MSR.cu -o executable

Run Using:
./executable ./JH_2_HCD <xx> 1
xx = sampling percentage e.g. 10, 20, 30