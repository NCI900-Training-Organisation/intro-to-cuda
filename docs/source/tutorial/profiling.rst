Profile using Nsight Systems
==============================

.. admonition:: Overview
   :class: Overview

    * **Time:** 30 min

    * Learn how to use **nsys** for performance analysis.
    * Learn how to generate detailed reports for optimization.


NVIDIA Nsight Systems (nsys)
--------------------------------

``nsys`` is NVIDIA's system-wide performance analysis tool used to profile CUDA applications and 
understand how your GPU and CPU interact. It provides timeline-based visualization and runtime 
statistics for:

* GPU kernel launches and memory transfers
* CPU threads and system calls
* CUDA runtime/API calls


To profile the code first compile with line info

.. code-block:: bash
    :linenos:

    nvcc -O2 -lineinfo -o 17_vector_add_unified 17_vector_add_unified.cu


then profile the executable

.. code-block:: bash
    :linenos:

    nsys profile \
        --stats=true \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        -o 17_vector_add_unified_profile \
        ./17_vector_add_unified


.. list-table:: Explanation of `nsys profile` flags
   :widths: 20 80
   :header-rows: 1

   * - Option
     - Description
   * - ``--stats=true``
     - Prints a summary of profiling statistics (e.g., kernel execution time, memory transfers) in the terminal after profiling.
   * - ``--trace=cuda,nvtx,osrt``
     - Enables tracing for selected domains:
       - ``cuda``: CUDA API calls and kernel activity
       - ``nvtx``: User-defined NVTX ranges/markers
       - ``osrt``: OS runtime info (e.g., CPU threads, scheduling)
   * - ``--cuda-memory-usage=true``
     - Reports CUDA memory usage, including allocation and deallocation across the application.
   * - ``-o ${exe}_profile``
     - Sets the base name for output files. Generates:
       - `${exe}_profile.qdrep`: Profiling report (open in `nsys-ui`)
       - `${exe}_profile.sqlite`: Structured performance data
   * - ``./${exe}``
     - Runs the compiled executable being profiled.


.. admonition:: Key Points
   :class: hint

    * NVIDIA Nsight Systems (`nsys`) is a tool for profiling CUDA applications, providing insights into GPU and CPU interactions.
    * Compile your CUDA code with `-lineinfo` for detailed profiling information.
    * Use `nsys profile` with appropriate flags to collect execution statistics, trace CUDA and system activity, and report memory usage.
    * Profiling results include summary statistics and detailed reports for performance analysis and optimization.