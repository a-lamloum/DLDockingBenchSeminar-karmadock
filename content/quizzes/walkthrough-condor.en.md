# quiz: walkthrough-condor

## purpose

### Q1
Why are the submit files not considered implementations of the machine-learning algorithms?

### A1
They configure execution conditions and invoke wrappers; model training, docking, conversion, and scoring logic lives in the Python and shell programs they launch.

### Q2
What problem does keeping job resources and file staging in the submit files solve?

### A2
It makes each cluster job declare the environment, inputs, outputs, hardware needs, and logs needed to reproduce that execution separately from the program logic.

### Q3
Why are there separate submit files for prototype, full-data, inference, evaluation, and Stage-2 experiments?

### A3
Those jobs require different wrappers, inputs, checkpoints, resources, persistence strategies, and output routes even though they share the same container image.

## how-it-fits-in

### Q1
What happens between `condor_submit` and execution of a repository wrapper?

### A1
HTCondor matches a worker, creates a temporary sandbox, stages declared inputs, starts the wrapper inside the Docker image, and later transfers declared outputs and logs back.

### Q2
Why must `condor/evaluate.sub` run after the inference jobs?

### A2
Its inputs are the pose directories produced by inference, so it cannot construct the prototype scoring matrix until those predictions exist.

### Q3
How do the inference, training, DDP, and evaluation submit files connect to the lower orchestration layer?

### A3
They call `run_infer.sh`, a training wrapper, a Stage-2 DDP launcher, or `evaluate.sh` respectively, passing the arguments and environment each wrapper expects.

## walkthrough

### Q1
Why does `queue 1` not imply that a DDP job uses only one GPU process?

### A1
It creates one Condor job, but that job can reserve multiple GPUs and let `torchrun` start one local process per GPU inside the single sandbox.

### Q2
Why do full-data and DDP jobs rely on a mounted home instead of transferring all inputs and checkpoints?

### A2
Their data and graph artifacts are large and their checkpoints must persist during long runs, so the mounted home avoids repeated transfers and `ON_EXIT` persistence risk.

### Q3
How do the two-GPU final job and the smoke job use resource settings for different goals?

### A3
The final job constrains capable high-memory GPUs and sets accumulation for effective batch 64, while the three-epoch smoke job accepts effective batch 32 because it tests launch, NCCL, sharding, reductions, and checkpointing rather than final training quality.

## gotchas--notes

### Q1
Why can submitting from the wrong directory break an otherwise valid `.sub` file?

### A1
Relative submit paths are resolved from the directory where `condor_submit` runs, and different files assume either the repository root or a deployment directory with bare staged files.

### Q2
Why is `ON_EXIT` insufficient protection for prototype checkpoints during a long interrupted job?

### A2
It transfers outputs only after normal job termination, so checkpoints inside the sandbox may not be copied continuously or survive ordinary interruption and rescheduling behavior.

### Q3
Why can `posebusters_infer.sub` fail before docking even though its staging directives are coherent?

### A3
The staged CSV uses `ligand_file`, but `seminar_csv.complex_records` recognizes `ligand_file_name` or the four-column full schema and can raise a pandas missing-column error.
