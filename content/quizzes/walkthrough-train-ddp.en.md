# quiz: walkthrough-train-ddp

## purpose

### Q1
How does DDP divide work differently from single-process `DataParallel`?

### A1
DDP runs one process and model replica per GPU on disjoint complex subsets, then collectively synchronizes gradients so every replica applies the same update.

### Q2
Why must metrics be explicitly reduced even though DDP already synchronizes gradients?

### A2
Gradient synchronization keeps model parameters aligned, but each process observes local losses and counts that must be combined to report correct global means and make identical stopping decisions.

### Q3
What happens when the file is launched with plain `python3` instead of a multi-process `torchrun` environment?

### A3
`WORLD_SIZE` defaults to one, distributed setup is skipped, and the trainer follows its compatible single-process path.

## how-it-fits-in

### Q1
How do the Stage-2 launchers decide whether to invoke this trainer with `torchrun`?

### A1
They count visible GPUs and use `torchrun` only when more than one is visible; otherwise they call the same script with ordinary Python.

### Q2
Why does only rank 0 write checkpoints, CSV logs, and W&B data?

### A2
All replicas should represent the same synchronized model, so one writer avoids duplicate runs and file races while other ranks continue participating in collectives.

### Q3
Why does this DDP variant omit the single-process trainer's `--jitter` option?

### A3
The implemented interface relies on on-the-fly dataset pose randomization and does not expose an additional training-loop coordinate perturbation.

## walkthrough

### Q1
How does manual ID striping give ranks disjoint data, and what property does it not provide?

### A1
Rank `r` takes every `world_size`-th ID starting at `r`, creating disjoint fixed membership, but it does not reshuffle membership across epochs like `DistributedSampler`.

### Q2
Why are validation sums and counts all-reduced instead of averaging each rank's mean?

### A2
Ranks can have unequal numbers of valid samples, so reducing totals and denominators avoids giving a small shard the same weight as a larger shard.

### Q3
Why must `find_unused_parameters=True` be enabled for Stage 1?

### A3
With `pos_r=0`, the docking branch can leave parameters without gradients; unused-parameter detection prevents DDP from waiting for reductions that will never occur.

### Q4
How are checkpoint key prefixes handled across bare models, `DataParallel`, and DDP?

### A4
Initialization strips one leading `module.` before loading the bare model, and wrapping then restores the prefix expected by DDP or the compatible single-process `DataParallel` format.

### Q5
Why can every rank break early stopping at the same epoch even though only rank 0 saves files?

### A5
All-reduced validation loss gives every rank the same improvement signal and manual patience state, while synchronized gradients keep model replicas equivalent.

## gotchas--notes

### Q1
Why may the current `no_sync()` block fail to suppress intermediate gradient communication?

### A1
PyTorch requires the forward pass to occur inside the context, but this source places only `backward()` inside it.

### Q2
How can fixed striping and identical seeds reduce augmentation diversity across ranks?

### A2
Each rank always owns the same subset and starts equivalent random streams, so rank-independent augmentation operations can become correlated even though samples differ.

### Q3
Why does setting `WORLD_SIZE>1` on a CPU-only machine fail rather than using a CPU distributed backend?

### A3
The distributed path unconditionally selects CUDA devices and the NCCL backend, both of which require supported NVIDIA GPU execution.
