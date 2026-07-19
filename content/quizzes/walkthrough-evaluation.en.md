# quiz: walkthrough-evaluation

## purpose

### Q1
Why does this evaluator produce one row per pose rather than only one summary row per complex?

### A1
Per-pose rows preserve rank, RMSD thresholds, paths, and optional validity checks, allowing top-1 and best-of-N summaries to be derived afterward.

### Q2
What two different aspects of docking quality do RMSD and PoseBusters evaluate?

### A2
RMSD compares predicted and reference ligand geometry, while PoseBusters checks whether a pose is physically and chemically plausible in its conditioning context.

### Q3
Why is symmetry awareness important when comparing ligand coordinates?

### A3
Chemically equivalent atom swaps should not penalize an otherwise identical pose, so the evaluator searches equivalent mappings for the minimum RMSD.

## how-it-fits-in

### Q1
How does the dataset name determine the evaluator's input locations?

### A1
It selects `data/<dataset>.csv`, reference structures under `data/<dataset>/`, and ranked predictions under `results/<dataset>/`.

### Q2
Why can `scripts/evaluate.sh` reuse this fixed directory interface for several prototype pipelines?

### A2
The wrapper creates temporary symbolic-link views that expose each pipeline's pose directory under the evaluator's expected `results/proto_test` name.

### Q3
What changes when `--no_pb_valid` is supplied?

### A3
PoseBusters checks are disabled, but ranked-pose loading, symmetry-corrected RMSD, aggregate statistics, and CSV output still run.

## walkthrough

### Q1
Why can `GetBestRMS` report an optimistic docking result for a ligand placed incorrectly in the protein pocket?

### A1
It optimally aligns the probe to the reference before measuring RMSD, removing global translation and rotation rather than preserving the fixed protein frame.

### Q2
What happens to reported rank when an early SDF record cannot be parsed?

### A2
Invalid records are skipped and the parseable list is compacted, so evaluated poses receive consecutive ranks that may differ from their original record positions.

### Q3
Why are global loss-style percentages derived from sums and result rows rather than treating missing complexes as failures?

### A3
The evaluator skips complexes that produce no pose rows; boolean means and medians are computed only over rows that reached evaluation, so coverage must be reported separately.

### Q4
How does best-of-N differ from the model's own top-1 ranking?

### A4
Best-of-N uses reference RMSD to choose the lowest-error pose among the first N, making it an oracle measure of whether a good candidate exists rather than whether the model ranked it first.

## gotchas--notes

### Q1
Why can PoseBusters validity become paired with the wrong RMSD pose?

### A1
PoseBusters processes the entire SDF in record order, while RMSD loading drops invalid records and stops at `top_n`, so positional pairing can diverge.

### Q2
How can skipped files inflate a reported success percentage?

### A2
Missing or unparseable complexes are removed from the denominator instead of counted as failures, leaving only complexes with result rows in the aggregate.

### Q3
Why is `pb_valid=None` not equivalent to a failed PoseBusters check?

### A3
It denotes unavailable check output and pandas normally excludes it from boolean means, whereas `False` represents an actual failed validity result.
