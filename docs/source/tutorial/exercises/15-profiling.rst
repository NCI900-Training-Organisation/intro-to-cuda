Exercise 15: Profiling with Nsight Systems
==========================================

.. admonition:: Exercise
   :class: todo

   **Time: 20 min**

      1. Run the code ``exercises/15-profiling/vector_add_unified.cu`` using `make & make profile`.
      2. Install `nsys <https://developer.nvidia.com/nsight-systems/get-started>`_ .
      3. Copy the file `vector_add_unified_profile.nsys-rep` to your laptop and inspect profile using `nsys`.

Submit a batch job
------------------

Compile this exercise separately with ``make`` first. From the repository root,
change to ``job-script`` and submit the prebuilt program on Gadi:

.. code-block:: bash

   cd job-script
   qsub 15-profiling.pbs

If you are already in ``job-script``, run only the ``qsub`` command.
See :doc:`../../job_submission` for setup, monitoring and output locations.
