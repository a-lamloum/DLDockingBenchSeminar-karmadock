# quiz: reference

## page

### Q1
Which parts belong to upstream KarmaDock, and which parts are the seminar team's work?

### A1
Upstream supplies the model, preprocessing and docking utilities, and released weights; the team supplies training, adapters, wrappers, Condor jobs, the Docker build, and analysis documentation.

### Q2
Why was reducing the requested GPU count a practical fix on the cluster?

### A2
Jobs requesting multiple GPUs remained idle for days, while the prototype could execute on one GPU, so the smaller request made scheduling feasible.

### Q3
What evidence in the repository helps distinguish authored code from reused upstream code?

### A3
The provenance tables enumerate the scripts and submission files the team created, explain which upstream modules they call, and state that no source inside upstream `KarmaDock/` was edited.
