# quiz: walkthrough-train

## purpose

### Q1
What capability does this file add that upstream KarmaDock does not provide?

### A1
It adds a complete single-process optimization, validation, checkpointing, resume, and logging loop around KarmaDock's existing model and forward losses.

### Q2
Why does the trainer combine RMSD-related and MDN losses rather than implementing a new docking architecture?

### A2
The upstream model already performs encoding and recycled coordinate refinement; this trainer supplies the optimization protocol and uses `pos_r` to control how its existing losses contribute.

### Q3
Why are both a best checkpoint and a resume checkpoint written?

### A3
`karmadock_team002.pkl` captures the best validation model for inference, while `last.pt` retains the latest model, optimizer, epoch, and patience-related state needed to continue training.

## how-it-fits-in

### Q1
How do the prototype and full-data wrappers use the same trainer for different data policies?

### A1
`run_train.sh` can request a deterministic random validation fraction, while `run_full_train.sh` supplies separate curated validation CSV and graph paths.

### Q2
Why must `--csv`, `--graph_dir`, and `--out_dir` come from the orchestration layer?

### A2
They identify the complex population, precomputed graph location, and persistent experiment state without embedding job-specific paths in the training loop.

### Q3
Which outputs allow the run to be inspected without W&B?

### A3
The local `train_log.csv`, best checkpoint, and `last.pt` remain available even when W&B is disabled, unavailable, or operating offline.

## walkthrough

### Q1
Why does graph availability filtering happen before splitting or loader construction?

### A1
Some structures fail preprocessing, so filtering first prevents missing `.dgl` files from crashing later and ensures train/validation lists contain only loadable graph stems.

### Q2
How do on-the-fly loading and coordinate jitter provide different forms of augmentation?

### A2
KarmaDock's dataset can randomize the ligand starting pose whenever a graph is loaded, while jitter explicitly adds small Gaussian coordinate perturbations inside the training loop.

### Q3
Why does the script use `DataParallel` with only GPU 0?

### A3
It does not gain multi-GPU scaling, but it preserves the `module.`-prefixed parameter format expected by upstream inference and related checkpoints.

### Q4
What sequence occurs at a complete gradient-accumulation boundary?

### A4
After scaled losses have accumulated, gradients are clipped, Adam updates the parameters, and gradient buffers are cleared before the next window.

### Q5
Why are `last.pt` and the best-validation checkpoint updated by different rules?

### A5
`last.pt` is saved every epoch for recoverability, whereas the early stopper replaces the inference checkpoint only when validation loss improves.

## gotchas--notes

### Q1
Why can resuming change the effective early-stopping schedule?

### A1
Resume restores the best score but does not explicitly restore the upstream stopper's count of consecutive non-improving epochs.

### Q2
What risk accompanies `strict=False` when loading an initialization checkpoint?

### A2
It tolerates related checkpoint formats but can silently ignore missing or unexpected model keys because the script does not report those mismatches.

### Q3
Why can a very small usable graph set produce invalid training behavior without a clear early error?

### A3
The split forces at least one validation ID, so one available graph can leave no training IDs, and empty loaders are not explicitly rejected.
