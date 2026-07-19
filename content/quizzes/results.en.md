# quiz: results

## page

### Q1
Why should the align-corrected success rate not be read as the model's raw docking accuracy?

### A1
Alignment superimposes the predicted ligand onto the reference frame, so it applies reference-dependent post-processing that the uncorrected output does not receive.

### Q2
What does the headline `success@2Å` metric count?

### A2
It is the fraction of top-ranked poses whose symmetry-corrected RMSD is below 2 Å, represented by rows with `pose_rank == 1` and `rmsd_lt2 == True`.

### Q3
Which prototype result is the primary submission result, and where is its per-complex data stored?

### A3
The primary prototype result is the uncorrected P2 from-scratch output, with per-complex values in `results/proto_test_evaluation.csv`.
