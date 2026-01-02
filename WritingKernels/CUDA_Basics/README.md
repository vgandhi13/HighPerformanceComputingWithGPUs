1. In a typical cuda program: we allocate something in cpu memory (malloc). Once there, we transfer to gpu memory. Once there cpu launches kernel (a gpu function that can be parallelized) on gpu. CPU then copies result from cpu back to gpu. cpu is the host which runs functions, gpu is the device which runs kernels.a function that is invoked for execution on the GPU is, for historical reasons, called a *kernel*.
2. A kernel launch can be thought of as starting many threads executing the kernel code in parallel on the GPU. GPU threads operate similarly to threads on CPUs, though there are some differences
3. When app launches kernel, it does so with millions of threse which are organized in blocks called thread blocks which in turn are organized into a grid. All threads of a thread block executed in a single Streaming multiprocessor (SMs) which allows them to communicate and synchronize efficiently. Threads within. a thread block have acces to the on chip shared memory which can be used to access information between threads of thread block.
4. A grid may consist of millions of thread blocks, while the GPU executing the grid may have only tens or hundreds of SMs
5.  There is no guarantee of scheduling between thread blocks, so a thread block cannot rely on results from other thread blocks, as they may not be able to be scheduled until that thread block has completed
6. The CUDA programming model enables arbitrarily large grids to run on GPUs of any size, whether it has only one SM or thousands of SMs. To achieve this, the CUDA programming model, with some exceptions, requires that there be no data dependencies between threads in different thread blocks. Different thread blocks within the grid are scheduled among the available SMs and may be executed in any order. 

## WARPS and SIMT
1. Within a thread block, threads are organized into groups of 32 threads called warps
2.  A warp executes the kernel code in a Single-Instruction Multiple-Threads (SIMT) paradigm. In SIMT, all threads in the warp are executing the same kernel code, but each thread may follow different branches through the code. That is, though all threads of the program execute the same code, threads do not need to follow the same execution path.

## Memory
1. DRAM attached to GPUis global memory because all SMs have access to it.
2. Each GPU has some on chip memory. Each SM has its own register file and shared memory which can be accessed quickly only by threads executing on SM.
3. Each SM has L1 cache. Larger L2 cache shared by all SMs within GPU.

## Compilation
1. CUDA applications and libraries are usually written in a higher-level language like C++. That higher-level language is compiled to PTX, and then the PTX is compiled into real binary for a physical GPU, called a CUDA binary, or cubin for short. A cubin has a specific binary format for a specific SM version, such as sm_120.
2. The GPU code is stored within a container called a fatbin. When an application is run, its GPU code is loaded onto a specific GPU and the best binary for that GPU from the fatbin is used.
Cubin code compiled for compute capability 8.6 will not load on GPUs of compute capability 9.0.
3. NVCC (NVIDIA CUDA Compiler) separates your code into "Host code" (CPU) and "Device code" (GPU). It sends the CPU code to a standard compiler like gcc or cl.exe and compiles the GPU code itself into a format called PTX
4. NVRTC (NVIDIA Runtime Compilation)
nvrtc is a library that allows you to compile CUDA C++ source code to PTX while your program is already running.


##  Kernels
1. As mentioned in the introduction to the CUDA Programming Model, functions which execute on the GPU which can be invoked from the host are called kernels. Kernels are written to be run by many parallel threads simultaneously.
2. The code for a kernel is specified using the __global__ declaration specifier
3. This indicates to the compiler that this function will be compiled for the GPU in a way that allows it to be invoked from a kernel launch. A kernel launch is an operation which starts a kernel running, usually from the CPU. Kernels are functions with a void return type.
4. The number of threads that will execute the kernel in parallel is specified as part of the kernel launch.
5. There is a limit to the number of threads per block, since all threads of a block reside on the same streaming multiprocessor(SM) and must share the resources of the SM. On current GPUs, a thread block may contain up to 1024 threads. If resources allow, more than one thread block can be scheduled on an SM simultaneously.
6. the kernel will be setup for execution on the GPU, but the host code will not wait for the kernel to complete (or even start) executing on the GPU before proceeding