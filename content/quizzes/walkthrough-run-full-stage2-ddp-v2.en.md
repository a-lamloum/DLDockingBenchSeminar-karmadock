# quiz: walkthrough-run-full-stage2-ddp-v2

## purpose

### Q1
Why was a second launcher created instead of editing the original Stage-2 DDP script?

### A1
Keeping both files preserves the historical experiment and makes the two hyperparameter changes explicit and auditable.

### Q2
Which choices make the v2 launcher more paper-faithful according to its documented rationale?

### A2
It sets weight decay to `0` and early-stopping patience to `70` instead of `1e-4` and `20`.

### Q3
Why is holding the remaining command line constant important?

### A3
It isolates the effect of weight decay and patience rather than confounding the comparison with different data, objectives, initialization, or batch behavior.

## how-it-fits-in

### Q1
How does `full_stage2_2gpu.sub` preserve global effective batch 64 with two GPUs?

### A1
It supplies per-process batch 8 and accumulation 4, giving `2 × 8 × 4 = 64`.

### Q2
Which operational behaviors are intentionally shared with the original launcher?

### A2
Paths, environment overrides, Stage-1 initialization, W&B setup, resume directory, and the `torchrun` versus plain-Python branch remain the same.

### Q3
Why does the submit file point explicitly to the v2 launcher?

### A3
The executable selection makes the final two-GPU job use the revised patience and weight-decay settings without altering other DDP jobs.

## walkthrough

### Q1
Why is no extra `--jitter` argument passed by this launcher?

### A1
The documented design relies on KarmaDock's on-the-fly dataset loading to randomize starting ligand poses rather than adding another coordinate perturbation here.

### Q2
What are the only meaningful differences in the v2 `ARGS` array?

### A2
It uses `--weight_decay 0` and `--patience 70`; the combined Stage-2 objective, data, initialization, and remaining optimization arguments stay fixed.

### Q3
Why is the printed checkpoint described as the best model rather than the last epoch?

### A3
Training saves `karmadock_team002.pkl` on validation improvement, so early stopping or later degradation can make it differ from final-epoch state.

## gotchas--notes

### Q1
Why do the original launcher's GPU-count and batch-size caveats still apply?

### A1
The v2 file copies the same GPU detection, process selection, environment defaults, and resume structure; only two trainer hyperparameters differ.

### Q2
What does the phrase “paper-faithful” guarantee programmatically?

### A2
Nothing beyond the recorded command-line choices; the repository does not automatically compare those values with the paper.

### Q3
How can launching the original and v2 scripts interfere with each other?

### A3
Both default to the same `S2WORK/ckpt_s2_ddp` path, so one invocation can resume or overwrite state created by the other.
