# quiz: index

## page

### Q1
What is the central contribution of this seminar project beyond upstream KarmaDock?

### A1
It adds the missing reproducible training loop, data adapters, execution wrappers, and cluster orchestration while leaving the upstream model code unchanged.

### Q2
Why is keeping the upstream KarmaDock model and docking code unchanged important for the benchmark?

### A2
It keeps the comparison faithful to the published method, so differences can be attributed to the added training and evaluation workflow rather than a modified model.

### Q3
Which script is the project's main implementation artifact, and what two training modes does the surrounding workflow support?

### A3
The main artifact is `scripts/train.py`; the workflow supports two-stage training from scratch for P2 and fine-tuning released weights for P3.
