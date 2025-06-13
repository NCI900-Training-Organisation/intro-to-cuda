Finding Optimal GPU Occupancy
===================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 20 min

    #. Understand the concept of GPU occupancy.
    #. Learn how to calculate and optimize GPU occupancy.
    #. Use CUDA API functions to determine optimal block sizes and active blocks per multiprocessor.

GPU occupancy is a measure of how effectively the GPU's resources are utilized by a kernel. High occupancy 
can lead to better performance, 

.. important::

    Occupancy is not the only factor that determines performance; memory bandwidth and instruction throughput 
    also play significant roles.

``cudaOccupancyMaxPotentialBlockSize()`` 
---------------------------------------

``cudaOccupancyMaxPotentialBlockSize()``  is a CUDA API function that helps determine the maximum number of 
threads per block that can be launched on a GPU while maximizing occupancy. It calculates the optimal 
block size and the minimum number of blocks required to achieve maximum occupancy for a given kernel.

This function takes into account the kernel's resource usage, such as shared memory and registers, and
returns the maximum block size and the minimum number of blocks needed to achieve optimal occupancy.
It is particularly useful for optimizing kernel launches, as it helps developers choose the best block size
and grid configuration for their specific kernel and GPU architecture.

.. code-block:: c
    :linenos:

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,  // minimum grid size needed to achieve the best potential
        &blockSize,    // Block size
        vectorAdd2D,   // Kernel function
        0,             // Per-block dynamic shared memory usage intended, in bytes
        0)

``cudaOccupancyMaxActiveBlocksPerMultiprocessor()``
------------------------------------------------

``cudaOccupancyMaxActiveBlocksPerMultiprocessor()`` is a CUDA API function that calculates the maximum number
of active blocks that can be launched per multiprocessor on a GPU for a given kernel. It helps developers
determine the optimal number of blocks to launch in order to maximize GPU occupancy and performance.
This function takes into account the kernel's resource usage, such as shared memory and registers, and returns
the maximum number of active blocks that can be launched per multiprocessor for the specified kernel.

.. code-block:: c
    :linenos:

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,  // Number of blocks
        vectorAdd2D, // Kernel function
        blockSize,   // Block size
        0)           // Per-block dynamic shared memory usage intended, in bytes



.. admonition:: Key Points
   :class: hint
   
    - GPU occupancy is a measure of how well the GPU's resources are utilized by a kernel.
    - High occupancy can lead to better performance, but it is not the only factor.
    - Use `cudaOccupancyMaxPotentialBlockSize()` to find the optimal block size for maximum occupancy.
    - Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` to determine the maximum number of active blocks per multiprocessor.
    - Consider other factors like memory bandwidth and instruction throughput when optimizing performance.