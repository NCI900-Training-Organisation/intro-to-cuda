
Understanding Warps in CUDA
==============================

.. admonition:: Overview
   :class: Overview

    * **Time:** 30 min

    #. Learn about warps in CUDA programming.
    #. Understand how warps affect performance and execution in CUDA kernels.


Warps are a fundamental concept in CUDA programming, representing how threads are organized and executed on NVIDIA GPUs. Understanding warps is crucial 
for writing efficient CUDA kernels. 

What is a Warp?
------------------
A warp is a group of threads that execute the same instruction at the same time on a CUDA-enabled GPU.

* A **warp** consists of **32 threads** (in all current GPUs) that are executed **in parallel**.
* When a CUDA kernel is launched, threads are grouped into **blocks**, which are further divided into **warps** of 32 threads each.
* The 32 threads in a warp execute the **same instruction simultaneously** (SIMT - Single Instruction, Multiple Threads), but on **different data**.

Why Use Warps?
------------------

Warps are designed to:

* Maximize **instruction-level parallelism**.
* Efficiently utilize the GPU's **SIMD-style execution units**.
* Enable lightweight and efficient scheduling using the **warp scheduler**.

Thread Grouping into Warps
------------------------------------

If a thread block contains 128 threads:

::

    128 threads / 32 threads per warp = 4 warps

* Threads 0–31 form warp 0.
* Threads 32–63 form warp 1.
* Threads 64–95 form warp 2.
* Threads 96–127 form warp 3.

Execution Behavior
------------------

* **Threads in a warp execute in lockstep**, following the same instruction.
* If threads diverge in control flow (e.g., due to ``if`` conditions), **warp divergence** occurs, causing **serialized execution** and reduced performance.

Warp Divergence
------------------

Warp divergence happens when threads in the same warp **follow different execution paths**.

Example:

.. code-block:: cpp

    int tid = threadIdx.x;
    if (tid % 2 == 0) {
        // Even threads do this
    } else {
        // Odd threads do that
    }

In this example, even and odd threads follow different branches, leading to warp divergence. This means that the warp will execute the even threads first, 
then the odd threads, effectively serializing the execution and reducing performance.


Similaly it is a bad parctice to have threads numbers that are not a multiple of 32 in a warp. This can lead to inefficient use of resources and increased 
execution time.

For example, if a CUDA thread block has 30 threads, the last two threads in the warp will not be utilized, leading to wasted resources.
If a CUDA thread block has 40 threads, the block will be split as follows:

* Warp 0: threads 0–31
* Warp 1: threads 32–39 (only 8 threads)

So, the block will contain 2 warps, but the second warp is partially full (only 8 active threads).

* Warp 0 will execute normally with all 32 threads.
* Warp 1 will execute with only 8 active threads and the remaining 24 threads inactive (masked out).

The GPU still treats it as a full warp in terms of scheduling. Performance-wise, you're paying the cost of a full warp but using only part of it.


.. important::

    * Write control flow to **minimize warp divergence**.
    * Ensure that threads access **contiguous memory** for coalesced memory access.
    * Aim for all threads in a warp to **follow the same path** for maximum efficiency.

Some additinal details
------------------------------

* Each SM is divided into processing partitions (also called sub-partitions).

* Each partition contains its own warp scheduler and instruction dispatch unit, along with a portion of the execution pipelines.

* On Volta (V100), an SM has 4 processing partitions, each with a warp scheduler, for a total of 4 schedulers per SM.

* An SM can host up to 64 resident warps (2048 threads).

* In each cycle, up to 4 warps can issue instructions (one per scheduler), while the rest wait.

* This deep pool of resident warps allows the GPU to quickly swap in ready warps when others are stalled (e.g., on memory or synchronization), ensuring high utilization and latency hiding.


.. admonition:: Key Points
   :class: hint

    #. A warp is a group of 32 threads that execute the same instruction simultaneously.
    #. Warps are fundamental to CUDA's execution model, enabling parallel processing.
    #. Warp divergence can lead to performance issues; aim to minimize it.
    #. Threads should be organized to maximize coalesced memory access and minimize divergence.
    #. Understanding warps is essential for writing efficient CUDA kernels and optimizing performance.
