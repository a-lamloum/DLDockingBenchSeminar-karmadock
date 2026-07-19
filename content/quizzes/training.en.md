# quiz: training

## page

### Q1
Why is `pos_r` set to 0 in P2 Stage 1 and to 1 in Stage 2?

### A1
Stage 1 trains only the MDN interaction-distance objective; Stage 2 turns on the RMSD coordinate term so the model also refines ligand poses.

### Q2
How is the effective batch size of 64 obtained?

### A2
Each step uses `batch_size 4`, and gradients are accumulated for `accum_steps 16`, giving an effective batch of 4 × 16 = 64.

### Q3
Why is P2 Stage 1 validation RMSD reported as unavailable even though the log contains zeros?

### A3
Because `pos_r=0` disables coordinate training in that stage, the logged zero is not a meaningful RMSD measurement to optimize or compare.
