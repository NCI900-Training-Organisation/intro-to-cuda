Memory Pool
===================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 30 min

    #. Understand the concept of memory pools in CUDA.
    #. Learn how to create and manage memory pools using the CUDA API.
    #. Explore the benefits of using memory pools for performance optimization.

In CUDA, memory pools are a way to efficiently manage device memory allocations by reducing the 
overhead of frequent cudaMalloc and cudaFree calls. The memory pool gives you more control over 
how memory is allocated and reused on the GPU.

Traditional cudaMalloc and cudaFree are expensive operations. In workloads with frequent allocations
and deallocations this overhead can become a performance bottleneck. A memory pool is a pre-allocated 
chunk of memory from which smaller allocations are served. Instead of going to the OS or CUDA 
driver each time, memory requests are fulfilled from this pool. It allows for 

* Faster memory allocation/deallocation
* Better memory reuse
* Control over memory fragmentation

The mian differnce in implementation is that
* ``cudaMallocAsync()`` replaces ``cudaMalloc()``
* ``cudaFreeAsync()`` replaces ``cudaFree()``


Default Memory Pool
----------------------


The default memory pool in CUDA does not have a fixed size. Instead, it grows and shrinks 
dynamically based on allocation needs, up to the limits of the available device memory.

* The default pool starts empty.
* When you call cudaMallocAsync(), the pool requests memory from the system as needed.
* It can reuse memory from previous allocations if available.
* The pool will continue to grow until:
    - There is no more available device memory, or
    - A soft limit (like the release threshold) is reached and enforced by your settings.

When memory is freed using ``cudaFreeAsync()``, it is returned to the pool for reuse, rather than being
returned to the system immediately. This allows for faster subsequent allocations from the memory pool.


.. code-block:: c
    :linenos:

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float* d_ptr;
    size_t size = 1024 * sizeof(float);

    // Asynchronous allocation
    cudaMallocAsync((void**)&d_ptr, size, stream);

    // Use d_ptr in a kernel...

    // Asynchronous deallocation
    cudaFreeAsync(d_ptr, stream);

    cudaStreamDestroy(stream);


We can get the attributes of the default memory pool using the following code:

.. code-block:: c
    :linenos:

    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &current);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &high);

where ``cudaMemPoolAttrReservedMemCurrent`` gives the current size of the reserved memory in the 
pool, and ``cudaMemPoolAttrReservedMemHigh`` gives the maximum size of the reserved memory at any 
point in the pool.


Custom Memory Pool
----------------------

Custom memory pools allow you to create and manage your own memory allocator instead of relying on 
the default memory pool. This gives you more control over how and when memory is reserved and 
reused, which is useful in performance-critical or memory-constrained applications.

Custom memory pools allows to:

* Set allocation limits
* Track memory usage independently
* Control release thresholds
* Isolate subsystems or tasks using separate allocators


.. admonition:: Explanation
   :class: attention

    Control release thresholds in CUDA memory pools refer to settings that determine when the pool
    should release unused memory back to the system (device allocator).

Create a memory pool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: c
    :linenos:

    cudaMemPool_t myPool;
    cudaMemPoolProps props = {}; //struct that specifies the properties of the memory pool
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = 0; // device ID

    cudaMemPoolCreate(&myPool, &props);



The `cudaMemPoolProps` structure defines the properties for a custom CUDA memory pool. Below is a 
detailed explanation of each field:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Field
     - Description
   * - `allocType`
     - Specifies the type of memory to allocate. Options include:
       
       - `cudaMemAllocationTypeDevice`: Device memory (GPU global memory).
       - `cudaMemAllocationTypePinned`: Pinned host memory, page-locked.
       - `cudaMemAllocationTypeManaged`: Unified memory accessible by both host and device.
   * - `handleTypes`
     - Specifies how memory handles can be shared across processes. Options include:
       
       - `cudaMemHandleTypeNone`: No inter-process sharing.
       - `cudaMemHandleTypePosixFd`: Shareable via POSIX file descriptors (Linux).
       - `cudaMemHandleTypeWin32`: Shareable via Windows handles.
   * - `location.type`
     - Indicates the location type of the memory. Commonly set to:
       
       - `cudaMemLocationTypeDevice`: Memory pool is tied to a specific GPU.
       - `cudaMemLocationTypeHost`: Host-based memory pool (rare).
   * - `location.id`
     - Specifies the device or host ID. For device memory pools, this is the GPU ID (e.g., `0` for `cudaSetDevice(0)`).


Set attributes (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Attribute

.. code-block:: c
    :linenos:

    cudaMemPoolSetAttribute(myPool, cudaMemPoolAttrReleaseThreshold, 1024 * 1024); // 1 MB threshold
    cudaMemPoolSetAttribute(myPool, cudaMemPoolAttrReservedMemCurrent, 512 * 1024 * 1024); // 512 MB reserved
    cudaMemPoolSetAttribute(myPool, cudaMemPoolAttrReservedMemHigh, 1024 * 1024 * 1024); // 1 GB high limit
    

The following attributes are configured for a custom memory pool using `cudaMemPoolSetAttribute`. Each attribute influences the behavior of memory allocation, reuse, and release.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Attribute
     - Description
   * - `cudaMemPoolAttrReleaseThreshold = 1024 * 1024`
     - Sets the maximum number of unused bytes (1 MB) the memory pool can retain before it begins releasing memory back to the system.
   * - `cudaMemPoolAttrReservedMemCurrent = 512 * 1024 * 1024`
     - (Optional/Advanced) Suggests setting the current reserved memory to 512 MB. Not always user-configurable—used more for querying.
   * - `cudaMemPoolAttrReservedMemHigh = 1024 * 1024 * 1024`
     - Sets a soft cap (1 GB) for the high watermark of memory usage within the pool, useful for monitoring purposes.
   

.. admonition:: Explanation
   :class: attention

    * The high watermark is the highest amount of memory the pool has ever allocated or reserved at any point in time.
    * This attribute records or sets a soft limit of 1 GB as that peak usage.
    * It doesn't enforce a strict limit but serves as a reference point to monitor or track how much memory the pool is using at its peak.
    * This can help developers understand memory usage patterns and detect if memory consumption approaches or exceeds expected values.


Use the memory pool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

cudaMallocFromPoolAsync() allocates memory from the specified memory pool instead of the default device 
memory allocator.

.. code-block:: c
    :linenos:

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    void* ptr;
    cudaMallocFromPoolAsync(&ptr, size, myPool, stream);


Pool trimming 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pool trimming in CUDA memory management refers to the process where the memory pool releases unused 
memory back to the operating system or underlying system allocator.

* Over time, the pool can accumulate unused memory chunks that are no longer needed by the application.
* Pool trimming is the act of freeing these unused memory chunks, reducing the memory footprint of the pool.
* This helps in controlling memory usage and preventing the application from holding excessive unused memory.


.. code-block:: c
    :linenos:

    cudaMemPoolTrimTo(myPool, releaseThreshold); // Trim the pool to release memory below the threshold of 1 MB

.. admonition:: Explanation
   :class: attention

    The ``releaseThreshold`` is amount of unused memory (in bytes) you want the pool to release back to the 
    system. This controls how aggressively the pool trims unused allocations

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Feature
     - Default Memory Pool
     - Custom Memory Pool
   * - Global shared pool
     - Yes
     - No (per-instance)
   * - Automatically initialized
     - Yes
     - No
   * - Can be configured
     - Partially (release only)
     - Fully (limits, thresholds)
   * - Used by ``cudaMallocAsync``
     - Yes
     - No (must use ``cudaMallocFromPoolAsync``)
   * - Lifetime
     - Tied to context/device
     - You manage it



.. admonition:: Key Points
   :class: hint

    - Memory pools in CUDA allow for efficient memory management by reducing allocation overhead.
    - The default memory pool grows dynamically and reuses memory for faster allocations.
    - Custom memory pools provide more control over allocation limits, reuse policies, and release thresholds.
    - Use `cudaMallocFromPoolAsync()` to allocate from a custom memory pool.
    - Pool trimming helps manage memory usage by releasing unused chunks back to the system.

