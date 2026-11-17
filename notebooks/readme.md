# Notebook workflow

1. Run the setup cell to locate the checkout. The notebooks work from the repository
   root, `notebooks`, or the default Gadi checkout under `/scratch/vp91/$USER`.
2. Open `02-gpu.ipynb` and run its build cell once before submitting any exercise,
   including Exercise 01. Have GCC, CUDA, Intel MKL, CMake and Ninja available to
   the kernel. Return to this cell to rebuild after changing CUDA source.
3. Follow each notebook's exercise steps. Command-line instructions use notebook `!` commands. The submission cells use the documented
   PBS scripts from `job-script`, followed by `qstat`. They submit jobs rather than
   executing GPU binaries in the notebook kernel.
4. Use monitoring cells to check again. Do not rerun a submission cell just to
   monitor an existing job. Read the specific job's PBS log after completion, then
   rerun its result/visualisation cells.

Source files are linked with `target="_blank"` so they open in a new tab, instead
of displaying the entire file inline. Images are displayed using Python
`Image`/`SVG` calls. Exercise 03 renders
`draw-indices.py` inline from notebook variables without a GPU. Its four GPU step jobs are submitted separately in teaching order.

Warps and brightness results are read from the files created in their exercise
folders by the PBS jobs. The shared-memory notebook's preview settings do not
override the batch job's parameters; edit the job script to change its GPU inputs.
The unified-memory comparison submits the same documented job before and after
editing/rebuilding the original prefetch calls.

Derived notebook images are saved under `notebook-output`. Saved execution outputs
are cleared so the notebooks do not show results from an earlier source version.
