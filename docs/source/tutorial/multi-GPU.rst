Multi-GPU Workflow
==================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 20 min

    #. Learn how to use multi-GPU programming.
    #. Understand the setup and synchronization of multiple GPUs.


cudaGetDeviceCount
----------------------

``cudaGetDeviceCount`` is used to determine how many CUDA-capable GPUs are available on your system.

.. code-block:: c
    :linenos:

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);


cudaSetDevice
-------------------


``cudaGetDeviceCount`` is used to determine how many CUDA-capable GPUs are available on your system.


.. code-block:: c
    :linenos:

    cudaSetDevice(0);
    cudaMalloc((void**)&d_A, dataSize);

    cudaSetDevice(1);
    cudaMalloc((void**)&d_B, dataSize);


In the code above `d_A` will be allocated on GPU-0 while `d_B` will be allocated on GPU-1.

cudaMemcpyDeviceToDevice
--------------------------

``cudaMemcpyDeviceToDevice`` flag will allow peer-to-peer ``cudaMemcpy`` between GPUs.

.. code-block:: c
    :linenos:

    cudaMemcpy(d_C1, d_C0, dataSize, cudaMemcpyDeviceToDevice);


.. admonition:: Key Points
   :class: hint

    * ``cudaGetDeviceCount()`` finds how many CUDA-capable GPUs are available on the system.
    * ``cudaSetDevice(device_id)`` selects the GPU on which subsequent operations (memory allocation, kernel launches) will occur.
    * ``Device-to-Device Copy`` flag enables device to device data copy.