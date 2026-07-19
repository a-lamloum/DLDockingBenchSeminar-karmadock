# quiz: walkthrough-run-infer

## purpose

### Q1
Why is this script described as a complete pose-generation path rather than an evaluation path?

### A1
It converts inputs, builds pockets and graphs, docks and scores poses, and exports predictions, but metric scoring is deliberately left to `evaluate.sh` or `evaluation.py`.

### Q2
Why are raw, force-field, and aligned outputs all exported?

### A2
They allow the benchmark to distinguish the network's direct prediction from two different postprocessing effects.

### Q3
What makes one invocation specific to a model experiment?

### A3
The dataset, checkpoint, and output tag arguments select the input structures, model weights, and result namespace.

## how-it-fits-in

### Q1
Why do inference submit files pass a checkpoint basename in some cases and an absolute path in others?

### A1
Transferred checkpoints arrive in the sandbox and are resolved against `$PWD`, while released checkpoints can be referenced directly inside the container image.

### Q2
What persistent output contract does the script provide to later scoring?

### A2
It writes evaluator-ready SDFs below `results/<tag>/<dataset>`, with `_ff` and `_align` sibling trees for corrected variants.

### Q3
Why are intermediate graphs not part of that output contract?

### A3
They exist only to support the current docking run inside a temporary directory and are removed at exit.

## walkthrough

### Q1
Why is a relative checkpoint resolved before changing into KarmaDock's utility directory?

### A1
Turning it into a sandbox-root path prevents later `cd` operations from changing what the relative model name refers to.

### Q2
What distinct work do `pre_processing.py`, `generate_graph.py`, and `ligand_docking.py` perform?

### A2
They identify and normalize pockets, serialize molecular relationships as graph tensors, and then predict, score, and correct ligand poses using the selected checkpoint.

### Q3
How does `export_variant` turn native repeat outputs into fair comparable result trees?

### A3
It invokes the same score-ranking converter with a different mode and suffix for each variant, producing best-first multi-record SDFs under parallel directories.

## gotchas--notes

### Q1
Why are temporary native docking outputs difficult to inspect after a failure?

### A1
The EXIT trap removes the temporary workspace on normal exits and most errors unless the script is changed to preserve or copy it.

### Q2
How can a diagnostic pose-count command itself stop the script?

### A2
It runs a glob and pipeline under strict `pipefail`; an empty result can produce a nonzero component even though diagnostics are redirected.

### Q3
Why should aligned success not be interpreted as reference-independent blind docking accuracy?

### A3
Alignment uses the reference pose or frame to superimpose coordinates, making it a postprocessing diagnostic with access to information unavailable to blind output.
