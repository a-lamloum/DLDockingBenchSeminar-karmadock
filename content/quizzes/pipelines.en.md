# quiz: pipelines

## page

### Q1
How do P1, P2, and P3 differ in where their weights come from?

### A1
P1 uses the released weights without training, P2 is trained from scratch with the two-stage protocol, and P3 fine-tunes the released weights.

### Q2
What determines whether the training workflow produces P2 or P3?

### A2
The initialization choice determines the path: P2 begins from scratch, while P3 supplies the released model through `--init_model` and fine-tunes it.

### Q3
Why does inference export three pose variants before evaluation?

### A3
It enables separate comparison of the raw prediction, force-field relaxation, and reference-frame alignment under the same official evaluator.
