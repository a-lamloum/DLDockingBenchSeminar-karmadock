# quiz: walkthrough-evaluate

## purpose

### Q1
Why does this wrapper evaluate a 3×3 matrix instead of calling the evaluator only once?

### A1
It applies the same dataset evaluator to three model pipelines and three pose variants so their RMSD CSVs are directly comparable.

### Q2
Why is the P2 raw CSV copied to `proto_test_evaluation.csv`?

### A2
That conventional filename is the repository headline result, while the explicitly named P2 source CSV remains available for provenance.

### Q3
Why is pose generation intentionally absent from this script?

### A3
Scoring is a separate reproducible stage that consumes existing results and can be rerun without repeating expensive docking.

## how-it-fits-in

### Q1
What must inference jobs produce before `condor/evaluate.sub` can use this wrapper?

### A1
They must create the available P1, P2, and P3 raw, FF, or aligned pose directories beneath `results/`.

### Q2
Why does the wrapper probe both repository-style and flattened locations?

### A2
HTCondor staging can preserve directory trees or place transferred inputs at the sandbox root, so both layouts are legitimate.

### Q3
What outputs can be absent without making the entire wrapper fail?

### A3
CSV outputs for missing pose directories and the headline copy when P2 raw results do not exist are skipped rather than treated as fatal.

## walkthrough

### Q1
Why does `eval_one` construct a temporary directory of symbolic links?

### A1
The Python evaluator expects fixed relative `data/proto_test` and `results/proto_test` paths, so symlinks present any selected pose tree through that interface without moving data.

### Q2
How does the `suffix:label` loop token produce both a directory path and an output filename?

### A2
Bash parameter expansion extracts the suffix before the colon for `proto_test${suffix}` and the label after the colon for the evaluation CSV name.

### Q3
Why is `--no_pb_valid` used for this batch comparison?

### A3
It keeps the matrix focused on RMSD and avoids repeating expensive PoseBusters checks across every pipeline and postprocessing variant.

## gotchas--notes

### Q1
Why can a failed evaluation leave temporary files behind?

### A1
Cleanup occurs only after a successful evaluator command and there is no trap covering interruption or failure inside `eval_one`.

### Q2
Why is exposing every pipeline as dataset `proto_test` correct rather than misleading?

### A2
The reference structures and CSV dataset stay the same; only the predictions selected through the temporary results symlink change.

### Q3
What happens to existing CSVs when this wrapper is rerun?

### A3
Successful evaluations overwrite their named result CSVs, and the P2 raw file overwrites the conventional headline copy.
