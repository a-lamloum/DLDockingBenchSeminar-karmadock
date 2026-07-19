# quiz: walkthrough-run-full-train

## purpose

### Q1
Why does full-data training use home-mounted storage instead of transferring all artifacts into each job sandbox?

### A1
The structures, generated graphs, checkpoints, and logs are large or resumable, so mounted persistent storage avoids repeated staging and loss at job exit.

### Q2
How is this script related to `run_train.sh`?

### A2
It implements the same single-process scratch and fine-tune training concepts for curated full-data splits, with persistence and path controls suited to the larger dataset.

### Q3
What limitation remains despite supporting the full dataset?

### A3
It still runs one GPU and does not exploit DDP, making full training functional but potentially impractical in wall time.

## how-it-fits-in

### Q1
Which mode does the listed Condor submit file use, and what other mode is available manually?

### A1
`condor/full_train_scratch.sub` invokes `scratch`; the wrapper also implements `finetune` even though no listed submit file selects it.

### Q2
Why are `full_train` and `full_val` supplied as separate inputs?

### A2
They preserve the curated validation split rather than randomly carving validation examples from the full training set.

### Q3
What does `$FULL_WORK_DIR` persist across jobs?

### A3
It holds converted structures, graphs, preprocessing markers, checkpoints, resume state, and W&B files for reuse.

## walkthrough

### Q1
Why does the `preprocess` function use a marker independently for each split?

### A1
Train and validation have separate converted structures and graph directories, so each expensive pipeline can be skipped only after its own successful completion.

### Q2
How does the `COMMON` Bash array improve correctness when building trainer commands?

### A2
It shares leak-resistant validation, batch, seed, resume, and logging arguments while preserving every path or value as a distinct shell word.

### Q3
How do scratch and fine-tune differ in their initialization and number of stages?

### A3
Scratch trains MDN-only Stage 1 from random weights and initializes combined Stage 2 from its best checkpoint; fine-tune starts from released weights and runs only one combined-loss stage.

## gotchas--notes

### Q1
What fails if the HTCondor job does not mount the user's home?

### A1
The default data and work paths point into the unavailable home mount, so large inputs, graphs, and persistent checkpoints cannot be found or written.

### Q2
Why can an invalid mode still consume substantial time before producing its error?

### A2
The script preprocesses both full-data splits before reaching the late `scratch` or `finetune` mode branch.

### Q3
Why can a `.preprocessed` marker become misleading?

### A3
It records prior completion but does not verify that underlying CSVs, structures, or graphs remain present, current, and uncorrupted.
