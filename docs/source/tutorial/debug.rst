Debug using cuda-gdb
=============================

.. admonition:: Overview
   :class: Overview

    * **Time:** 15 min

    * Learn how to use **cuda-gdb** for Debugging.


To debug a CUDA program, use cuda-gdb, NVIDIA’s debugger designed for CUDA C/C++ applications. 
It works similarly to gdb, but with additional support for GPUs.

To start compile the application with debug flags

.. code-block:: bash
    :linenos:

    nvcc -G -g -o my_program my_program.cu


.. admonition:: Explanation
   :class: attention

    * ``-G``: Includes debug information for device code
    * ``-g``: Includes debug info for host code (like gcc)

Then launch the application with ``cuda-gdb``

.. code-block:: bash
    :linenos:

    cuda-gdb ./18_vector_add


.. admonition:: Key Points
   :class: hint

   
    * Debugging CUDA code is similar to debugging CPU code with gdb, but includes GPU-specific features.
    * Always ensure you are running the debug build when using `cuda-gdb`.
