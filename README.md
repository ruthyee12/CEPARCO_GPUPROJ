# CEPARCO GPU PROJECT

## Overview
This project implements and benchmarks 1D convolution computation on a single input signal (array) using: C, CUDA. The execution time of each implementation is recorded and analyzed for performance comparison.

## Colab Link
- https://colab.research.google.com/drive/1cr4kM2cROON08fjmTapcvzKTbSEsDspZ?usp=sharing

## Video Recording

## Project Structure
### (1) C Program 
- Contains the C implementation of the 1D convolution computation
- Computes the correctness of the results and outputs onto console
- Used as a reference for correct output
### (2) CUDA (with prefetching, mem advse, and unified memory)
- Standard CUDA implementation 
- Since this was already proven to be faster than C, this will be used as the benchmark
### (3) CUDA with streams (with prefetching, mem adivse, and unified memory)
- Each kernel call of 1dconvolution function is divided into 4 streams
- Data is prefetched before and after
### (4) CUDA with streams and memcpy (no prefetching, mem advise, or unified memory)
- Data was memcpy'd once from host to device before the kernel calls and once after device to host
### (5) CUDA with streams and memcpy for each loope (no prefetching, mem advise, or unified memory)
- Memcpy from host to device was done before every kernel call while device to host was done after every kernel

## Screenshots of the Program Output with Execution Time and Correctness Check

- Figure 1: Screenshot of C Output with Execution Time and Correctness Check
<img src="screenshots/c_result.png">

-  Figure 2: Screenshot of CUDA Output with Execution Time and Correctness Check
<img src="screenshots/cuda_result.png">

-  Figure 3: Screenshot of CUDA (with streams) Output with Execution Time and Correctness Check
<img src="screenshots/cudastreams_result.png">

-  Figure 4: Screenshot of CUDA (with streams and memcpy) Output with Execution Time and Correctness Check
<img src="screenshots/4RESULTS.png">

-  Figure 5: Screenshot of CUDA (with streams and memcpy for each loope) Output with Execution Time and Correctness Check
<img src="screenshots/cudamemcpys_result.png">

## Screenshots of the Timeline Viewed through NVIDIA Nsight Systems

-  Figure 6: Timeline of Baseline CUDA
<img src="screenshots/2CUDABENCHMARK.png">

-  Figure 7: Timeline of (3)
<img src="screenshots/3CUDASTANDARD.png">

-  Figure 8: Timeline of (4)
<img src="screenshots/4CUDASTREAMBEFOREAFTER.png">

-  Figure 9: Timeline of (5)
<img src="screenshots/5CUDASTREAMMEMCPYALL.png">

## Comparison and Speedup
Speedup is computed as (2)/(X). Includes data transfer times. (X) refers to the Project Structure section of this document. 
- Speedup (2)/(3) : 1.04
- Speedup (2)/(4) : 0.99
- Speedup (2)/(5) : 0.07

- (2) = 335.86 ms + 89.1 ms + 79.91 ms = 504.87 ms
- (3) = 317.57 ms + 88.96 ms + 79.97 ms = 486.5 ms
- (4) = 338.47 ms + 87.63 ms + 81.95 ms = 508.05 ms 
- (5) = 330.48 ms + 3582 ms + 3559 ms = 7471.48 ms

## Results and Analysis

As we can see from the results and speedup computations, all the implementations have more or less the same times except for (5). Streams are like queues of processes and the best way to use them is to allocate memory that is unique to that stream. Similar to pipelining, each stream can execute processes concurrently if each stream uses different resources at a certain point in time. That means that if you allocate memory for a specific stream's use, the CPU or other streams should not be able to access to achieve concurrency. Our stream implementations partition each full 1D convolution into 4 streams, meaning each partition of the kernel receives 1/4 of the input and output arrays. However, due to 1D convolution needing in[i+1] and in[i+2], the problem of 1D convolution makes it inherently improbable to use with streams. Hence why all the stream implementations behave similarly to the benchmark CUDA even if a single 1D convolution on the full input array is partitioned into several streams. 

As for (5), that demonstrates how streams should look like. As soon as the initial H2D memcpy is finished, another H2D starts right after while the previous stream runs its kernel, achieving concurrency. D2Hs also occur while other streams do H2Ds. However, as seen from the timeline of (5), memcpys take too long as they are an older method of data transfer compared to prefetch and unified memory. While streams may not be as fast as unified memory, it is useful for when prefetching is not available such as on Windows. 



## Conclusion and Discussion

This project explored multiple CUDA implementations of 1D convolution and analyzed their performance relative to a baseline C program and a standard CUDA implementation. The results demonstrate that while streams can enhance performance in many GPU-accelerated applications, their benefits are limited for 1D convolution due to inherent data dependencies between adjacent elements. Prefetching, mem advise, and unified memory with the streams were able to provide slight performance boosts but do not yield significant improvements over the baseline CUDA implementation. 

Streams are more useful when prefetching is unavailable. Several problems when implementing this project were, for one, figuring out how streams work, analyzing the data, drawing conclusions on why the results are the way they are, and finding sufficient proof. Using the Nsight profiler provided very useful insights on if our streams were working correctly yet it was worrying to see how the timelines aren't concurrent. Several methods were tested to see if any could make the code faster. However, after multiple variations such as using with or without prefetch, mem advise, memcpy, etc., we found none could truly beat the prefetching time. In the end, the best result we could get was at least on par with the standard CUDA implementation. 

The concept of the default stream was also interesting as this means that any process allocated to the default stream will block all other streams from executing their processes. This is called implicit synchronization. As much as possible, we do not want to use the default stream if we want concurrency. 

As a side note, it was also interesting to learn the difference between pinned (memcpy) and managed memory (prefetch). With pinned memory, you have to manually allocate in and out arrays on both the CPU and GPU, pretty much doubling the amount of storage needed. Most stream implementations make use of pinned memory due to unified memory using implicit synchronization as memory allocation is done asynchronously by the processor, meaning data prefetching may occur at a "random" time, preventing kernels from accessing memory needed.

An important A-HA! moment we realized while creating the CUDA implementations is that in order to achieve the true concurrency behavior of streams is that we have to run all memcpyasynchs together with the kernel. This can be observed in the NSIGHT visualizer for CUDA (5). It was able to achieve the natural behavior of streams which allows kernels and data to be transferred on one stream without waiting for the others to finish. 

In conclusion, while CUDA streams offer potential benefits in some contexts, their effectiveness for 1D convolution is limited. Future optimizations for this could explore different ways of accessing and transferring memory, other parallelizations strategies, or using the shared memory concept to further enhance the performance of streams.
