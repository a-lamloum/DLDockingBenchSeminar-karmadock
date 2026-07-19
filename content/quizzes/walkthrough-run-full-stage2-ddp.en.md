# quiz: walkthrough-run-full-stage2-ddp

## purpose

### Q1
Why does this launcher reuse prebuilt graphs and a frozen Stage-1 checkpoint?

### A1
It isolates the multi-GPU Stage-2 experiment from expensive preprocessing and from changes to the main Stage-1 or full-training workspace.

### Q2
What makes the launcher's output isolated from its read-only inputs?

### A2
Inputs default to existing home-mounted data, graphs, and a snapshot, while checkpoints and W&B state are written below a separate `S2_WORK_DIR`.

### Q3
Why is DDP appropriate specifically for this full-data stage?

### A3
The large Stage-2 workload can distribute disjoint samples across GPUs while synchronizing model gradients, reducing wall time compared with one process.

## how-it-fits-in

### Q1
How do `full_stage2_mgpu.sub` and `smoke_2gpu.sub` alter the same launcher for different experiments?

### A1
They choose resources and environment overrides such as GPU count and epoch limit while staging the same launcher and DDP trainer.

### Q2
Why are the environment variables useful even though the script supplies defaults?

### A2
They relocate large home-mounted inputs and outputs or adjust batch, accumulation, checkpoint, and epoch settings without editing the executable.

### Q3
Which artifact is expected after a successful run?

### A3
The best-validation model is expected at `$S2_WORK_DIR/ckpt_s2_ddp/karmadock_team002.pkl`, with resume and logging state in the same isolated workspace.

## walkthrough

### Q1
How is the reported global effective batch computed?

### A1
It multiplies visible GPU processes by per-process physical batch `S2_BATCH` and accumulation windows `S2_ACCUM`.

### Q2
Why do `--pos_r 1` and explicit validation paths matter for Stage 2?

### A2
`pos_r=1` enables combined coordinate and scoring learning, while the separate full-validation graphs preserve the curated train/validation split.

### Q3
What changes when only one GPU is visible?

### A3
The launcher calls ordinary `python3` instead of `torchrun`, avoiding process-group setup while retaining the same trainer arguments.

## gotchas--notes

### Q1
Why might the GPU-count fallback never run if `nvidia-smi` itself fails?

### A1
The count is produced by a pipeline inside a `pipefail` script, so failure of `nvidia-smi` can terminate execution before the later zero-count fallback.

### Q2
How can a new W&B run name still resume old optimizer state?

### A2
The timestamp changes the displayed name, but `--resume` reads `last.pt` from the reused checkpoint directory independently of that name.

### Q3
Why do the default batch settings reproduce effective batch 64 only on four GPUs?

### A3
The defaults are batch 8 and accumulation 2, so the product reaches 64 only when `world_size` is four; other allocations need overrides.
