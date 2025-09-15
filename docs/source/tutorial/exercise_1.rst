Exercise 1
================



.. admonition:: Exercise
   :class: todo

   **Time: 20 min**

   Request a GPU on Gadi using the command 
   
   ``module load gcc/14.2.0 openmpi/4.1.7 cudnn/9.5.0-cuda12 intel-vtune/2024.2.1 cmake/3.8.2``
   ``module load cuda/12.9.0 papi/7.1.0 intel-mkl/2024.2.1``
   ``qsub -I -q gpuvolta  -P vp91 -l walltime=02:00:00,ncpus=12,ngpus=1,mem=96GB`` 
   
   and run the following code to check the GPU architecture.


    1. Check the GPU architecture of your system using the command ``nvidia-smi``.
    2. Run the code `exercises/exercise_1/1_architecture.cu` and check the output.`
