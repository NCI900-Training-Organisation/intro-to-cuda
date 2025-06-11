Unified Memory
================

.. admonition:: Overview
   :class: Overview

    * **Time:** 45 min

    #. Learn about Unified Memory in CUDA.
    #. Understand how Unified Memory simplifies memory management in CUDA applications.


Unified Memory is a memory management feature in CUDA (CUDA 6 and above)that allows developers to write applications without 
worrying about the complexities of managing memory between the host (CPU) and device (GPU). 
It provides a single address space for both the host and device, enabling seamless data sharing and access.

In Unified Memory we use ``cudaMallocManaged`` to allocate memory that is accessible from both the host and device.

.. code-block:: c
    :linenos:

    int *data;
    cudaMallocManaged(&data, size * sizeof(int));

    // Use data on the host
    for (int i = 0; i < size; i++) {
        data[i] = i;
    }

    // Use data on the device
    kernel<<<blocks, threads>>>(data);

    // Synchronize to ensure all operations are complete
    cudaDeviceSynchronize();

Under the hood, Unified Memory automatically migrates data between the host and device as needed.
This means that when the host accesses data that is currently on the device, Unified Memory will 
automatically transfer it to the host memory, and vice versa. This migration is managed by the CUDA runtime, 
which tracks memory accesses and performs the necessary transfers transparently.

The advantage of Unified Memory is:
* Simplifies programming — no need to manage explicit memory transfers
* Helps porting code from CPU to GPU
* Reduces the complexity of memory management

However, there are some considerations to keep in mind when using Unified Memory:

* May cause performance overhead due to page migration
* Less fine-grained control over memory movement
* Not all CUDA features are compatible with Unified Memory
* Limited support for certain data structures and algorithms


Pinned Memory in Unified Memory
-----------------------------

Managed Memory is the default type of Unified Memory allocation (using ``cudaMemAdvise``). It allows the CUDA 
runtime to automatically manage memory migration between the host and device. When you allocate managed memory,
the  CUDA runtime ensures that data is available on both the host and device as needed.

Another type of Unified Memory allocation is **pinned memory**. This type of Unified Memory allocation allows 
the host memory to be pinned, which means it cannot be paged out by the operating system. Pinned memory can 
improve performance for certain operations, such as asynchronous data transfers, but it requires more careful 
management.


.. code-block:: c
    :linenos:

    // Example of using Unified Memory with pinned memory
    __global__ void kernel(int *data) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        data[idx] += 1; // Increment each element by 1
    }

    int main() {
        int size = 1024;
        int *data;

        // Allocate Unified Memory with pinned memory
        cudaMallocHost(&data, size * sizeof(int));

        // Initialize data on the host
        for (int i = 0; i < size; i++) {
            data[i] = i;
        }

        // Launch kernel
        kernel<<<(size + 255) / 256, 256>>>(data);

        // Synchronize to ensure all operations are complete
        cudaDeviceSynchronize();

        // Free Unified Memory
        cudaFreeHost(data);
    }

There are some disadvantages to using pinned memory:

* Pinned memory is not pageable, which means it cannot be swapped out to disk by the operating system.
* It can consume more system memory, as it is not eligible for paging.
* It may lead to reduced system performance if too much pinned memory is used, as it can limit the amount of memory available for other processes.

cudaMemPrefetchAsync
-----------------------------

The ``cudaMemPrefetchAsync`` function is used to prefetch data from the device to the host or vice versa in a 
non-blocking manner. This can help improve performance by ensuring that data is available in the desired 
memory space before it is accessed.

.. code-block:: c
    :linenos:

    // Example of using cudaMemPrefetchAsync
    int *data;
    cudaMallocManaged(&data, size * sizeof(int));

    // Prefetch data to the host from GPU 0
    cudaMemPrefetchAsync(data, size * sizeof(int), cudaCpuDeviceId);

    // Use data on the host
    for (int i = 0; i < size; i++) {
        data[i] += 1;
    }

    // Prefetch data back to GPU 0
    cudaMemPrefetchAsync(data, size * sizeof(int), 0);

    // Launch kernel on the device
    kernel<<<blocks, threads>>>(data);

    // Synchronize to ensure all operations are complete
    cudaDeviceSynchronize();


.. admonition:: Explanation
   :class: attention

    ``cudaCpuDeviceId`` is a special device ID that refers to the host CPU.


cudaMemAdvise
-----------------------------

The ``cudaMemAdvise`` function is used to provide advice to the CUDA runtime about how memory should be managed.

.. code-block:: c
    :linenos:

    // Example of using cudaMemAdvise
    int *data;
    cudaMallocManaged(&data, N * sizeof(int));

    // Initialize data on host
    for (int i = 0; i < N; ++i)
        data[i] = i;

    // 1. Advise that data will be mostly read by the host (CPU)
    cudaMemAdvise(data, N * sizeof(int), cudaMemAdviseSetReadMostly, cudaCpuDeviceId);

    // 2. Prefer memory to be located on GPU 0
    cudaMemAdvise(data, N * sizeof(int), cudaMemAdviseSetPreferredLocation, 0);

    // 3. Specify that GPU 0 will access this memory
    cudaMemAdvise(data, N * sizeof(int), cudaMemAdviseSetAccessedBy, 0);



The different advices that can be provided using ``cudaMemAdvise`` include:

* ``cudaMemAdviseSetReadMostly``: Indicates that the memory will be read mostly by the host.
* ``cudaMemAdviseSetPreferredLocation``: Specifies the preferred location for the memory (host or device).
* ``cudaMemAdviseSetAccessedBy``: Indicates which device(s) will access the memory.

.. code-block:: c
    :linenos:

    // Example of using cudaMemAdvise with different advices
    int *data;
    cudaMallocManaged(&data, size * sizeof(int));

    // Advise the CUDA runtime that the data will be read mostly by the host
    cudaMemAdvise(data, size * sizeof(int), cudaMemAdviseSetReadMostly, 0);

    // Advise the CUDA runtime that the data will be accessed by device 0
    cudaMemAdvise(data, size * sizeof(int), cudaMemAdviseSetAccessedBy, 0);

    // Use data on the host
    for (int i = 0; i < size; i++) {
        data[i] += 1;
    }

    // Launch kernel on the device
    kernel<<<blocks, threads>>>(data);

    // Synchronize to ensure all operations are complete
    cudaDeviceSynchronize();


.. admonition:: Key Points
   :class: hint
   
    #. Unified Memory provides a single address space for both host and device memory.
    #. It simplifies memory management by automatically migrating data between host and device.
    #. Use ``cudaMallocManaged`` to allocate Unified Memory.
    #. Be aware of potential performance overhead due to automatic page migration.
    #. Not all CUDA features are compatible with Unified Memory, so check compatibility when using it.
    #. Pinned memory can improve performance for certain operations but requires careful management.
    #. Use ``cudaMemPrefetchAsync`` to prefetch data between host and device in a non-blocking manner.
    #. Use ``cudaMemAdvise`` to provide advice to the CUDA runtime about memory management.

