# quiz: walkthroughs

## how-these-files-fit-together

### Q1
Why does the pipeline need one converter before KarmaDock and another converter after docking?

### A1
The first converter changes seminar inputs into KarmaDock's naming and directory layout, while the second ranks KarmaDock repeats and restores the one-file-per-complex evaluator layout.

### Q2
How do `train.py` and `train_ddp.py` occupy different roles without changing the upstream KarmaDock model?

### A2
Both call the upstream model and losses, but `train.py` runs one process while `train_ddp.py` distributes disjoint complex stripes across one process per GPU and synchronizes gradients and metrics.

### Q3
Why are the files in `condor/` described as sitting above the `run_*.sh` wrappers?

### A3
The submit files choose resources, container, staging, arguments, outputs, and logs, then invoke shell wrappers that orchestrate the actual conversion, training, inference, or evaluation programs.

## walkthrough-index

### Q1
If a student wants to understand why graph stems must match CSV-derived complex IDs, which walkthrough should they read first?

### A1
They should start with `scripts/seminar_csv.py`, because its walkthrough defines the shared ID derivation used by converters and trainers.

### Q2
Which pair of walkthroughs best explains the transition from predicted KarmaDock poses to benchmark CSV metrics?

### A2
The prediction-layout converter explains ranking and SDF export, and `evaluation/evaluation.py` explains how those ranked SDF records become RMSD and PoseBusters result rows.

### Q3
Why does the index separate shell-driver walkthroughs from the Python training and evaluation walkthroughs?

### A3
The shell drivers explain reproducible orchestration and environment setup, while the Python pages explain data interpretation, model optimization, distributed synchronization, and metric computation.
