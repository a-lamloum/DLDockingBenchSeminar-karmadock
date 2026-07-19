# quiz: walkthrough-run-train

## purpose

### Q1
Why does the prototype training driver include preprocessing as well as trainer invocation?

### A1
It provides a self-contained route from seminar structures to KarmaDock graphs and then to P2 scratch or P3 fine-tune checkpoints.

### Q2
How do the two advertised modes correspond to the benchmark pipelines?

### A2
`scratch` runs MDN-only Stage 1 followed by combined Stage 2 for P2, while `finetune` starts from released weights and trains the combined objective for P3.

### Q3
Why is this wrapper described as single-process?

### A3
It always calls `train.py` directly and does not launch one process per GPU through `torchrun`.

## how-it-fits-in

### Q1
Which Condor files select each training mode?

### A1
`p2_train_scratch.sub` passes `scratch`, while `p3_finetune.sub` passes `finetune`.

### Q2
What must HTCondor stage into the current sandbox before this wrapper starts?

### A2
It needs `proto_train.csv`, the `proto_train/` structures, and the local `scripts/` directory, while upstream KarmaDock comes from the image.

### Q3
Which outputs are intended to survive the run?

### A3
Reusable work and graphs live under `work_train/`, and model checkpoints plus logs live under `ckpt/`, subject to the submit file's transfer behavior.

## walkthrough

### Q1
Why is `work_train` persistent and protected by a `.preprocessed` marker?

### A1
Rescheduled jobs can reuse costly converted structures and graphs, and the marker is written only after all strict-mode preprocessing commands succeed.

### Q2
How does Stage 1 obtain effective batch 64 without loading 64 complexes simultaneously?

### A2
It accumulates gradients over 16 physical batches of 4 before each optimizer update.

### Q3
Why does Stage 2 initialize from Stage 1 and change `pos_r` from 0 to 1?

### A3
Stage 1 learns MDN scoring first; Stage 2 reuses that best checkpoint and activates the RMSD coordinate term for docking refinement.

### Q4
How does fine-tuning differ from simply resuming Stage 2?

### A4
Fine-tuning initializes from the released screening checkpoint, skips the project-trained MDN-only stage, and writes to the separate P3 output directory with its own hyperparameters.

## gotchas--notes

### Q1
Why can marker files falsely indicate a complete training prerequisite?

### A1
They record that a previous command sequence finished but do not check whether graphs or checkpoints were later deleted, corrupted, or made stale.

### Q2
Why can a completed Stage 2 be invoked again even though Stage 1 is skipped?

### A2
Stage 1 has `stage.done`, but Stage 2 has no equivalent marker and relies only on optional `last.pt` resume behavior.

### Q3
Why is a final partial accumulation window underweighted?

### A3
Every microbatch loss is divided by 16 even when fewer than 16 batches remain before the forced final optimizer step.
