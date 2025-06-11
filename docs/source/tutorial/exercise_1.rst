Exercise 1
================



.. admonition:: Exercise
   :class: todo

   **Time: 20 min**

   Request a GPU on Gadi using the command ``qsub -I -q gpuvolta  -P vp91 -l walltime=02:00:00,ncpus=12,ngpus=1,mem=10GB`` and 
   run the following code to check the GPU architecture.


    1. Check the GPU architecture of your system using the command ``nvidia-smi``.
    2. Run the code `exercises/exercise_1/1_architecture.cu` and check the output.`
