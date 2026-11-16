# Gadi PBS job scripts

Compile each exercise separately with `make`. Then submit its job from this directory:

```bash
cd job-script
qsub 02-gpu-exec-model.pbs
```

There is one script for every exercise, numbered 00–16. Each script loads the
modules, changes to `../exercises/<exercise>`, and runs the prebuilt executable(s).
There are no build steps or shared shell helpers.

The defaults are project `vp91`, queue `gpuvolta`, 15 minutes, one GPU, 12 CPUs and
48 GB memory. Exercise 14 requests two GPUs, 24 CPUs and 96 GB. Change the PBS
project and storage lines if using another allocation or filesystem.

PBS logs appear in `job-script`, for example `02-gpu-exec-model.o123456`. Generated CSV,
SVG, raw-image and profiling files are saved in the corresponding exercise folder.
Repeated runs overwrite those generated files; run one job per exercise at a time.

- **00:** Requires the MKL module when compiling and running the CPU benchmark.
- **02:** Edit the four dimensions in both commands together.
  To run just the print exercises, submit `qsub 02a-print-launch.pbs` or
  `qsub 02b-print-global-index.pbs` from this directory after compiling manually.
- **04:** Writes `timestamps.csv` and prints its first 16 thread records.
- **06:** Generates `input.raw`, runs the brightness kernel and prints `stats.json`.
- **08:** Runs the version you compiled. Edit prefetch calls and rebuild to compare.
- **15:** Requires `nsys`; writes `vector_add_unified_profile.nsys-rep`.
- **16:** Uses `cuda-gdb` in batch mode to capture the intentional bug and backtrace.
