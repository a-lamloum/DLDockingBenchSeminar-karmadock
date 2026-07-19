# quiz: reproduce

## page

### Q1
What is the shortest path to verify the shipped P2 poses without using the cluster?

### A1
Unpack the reference structures, create the expected `data/proto_test` link, and run `python evaluation/evaluation.py --dataset proto_test` locally.

### Q2
Why can the HTCondor submission files run without site-specific path edits?

### A2
They transfer the required data and project code into each job and use the released weights already contained in the image.

### Q3
What is the difference between scoring the shipped poses and regenerating them?

### A3
Scoring only runs the evaluator over existing predicted structures; regeneration reruns preprocessing and docking for the pipelines before the evaluator is submitted.
