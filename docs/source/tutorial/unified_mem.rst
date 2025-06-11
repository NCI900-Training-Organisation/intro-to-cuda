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


.. admonition:: Key Points
   :class: hint
   
    #. Unified Memory provides a single address space for both host and device memory.
    #. It simplifies memory management by automatically migrating data between host and device.
    #. Use `cudaMallocManaged` to allocate Unified Memory.
    #. Be aware of potential performance overhead due to automatic page migration.
    #. Not all CUDA features are compatible with Unified Memory, so check compatibility when using it.

