# quiz: walkthrough-convert-seminar-to-karmadock

## purpose

### Q1
Why does this converter copy structures instead of modifying molecular chemistry?

### A1
The source molecules are already refined; upstream KarmaDock needs only a different per-complex directory and filename convention.

### Q2
What invariant does the converter establish for each usable complex?

### A2
It creates `<id>/<id>_ligand.sdf` and `<id>_protein.pdb`, the exact names expected by upstream preprocessing.

### Q3
Why must ligand and protein files be treated as a pair?

### A3
KarmaDock preprocessing needs both molecular partners, so a complex with only one source file cannot form a usable docking input.

## how-it-fits-in

### Q1
Why is this converter called before pocket extraction and graph generation?

### A1
Those upstream steps traverse KarmaDock's per-complex input layout, which does not match the seminar's flat structure layout.

### Q2
How does the shared `complex_records` helper reduce pipeline inconsistency?

### A2
It gives training and inference conversion the same ordered IDs and source filenames across supported CSV schemas.

### Q3
What are the roles of `--src_dir` and `--out_dir`?

### A3
`--src_dir` contains the seminar-named source structures, while `--out_dir` receives the reorganized KarmaDock directory tree.

## walkthrough

### Q1
Why are both input paths checked before the output directory is created?

### A1
The script fails clearly on a missing CSV or source directory before leaving partial output work behind.

### Q2
What happens when one molecular partner is absent for a CSV record?

### A2
The converter increments the missing count, skips that complex, and continues processing other complete pairs.

### Q3
Why is the final `converted` summary operationally important?

### A3
Missing pairs do not cause a nonzero exit, so the summary exposes silent coverage loss that downstream graph counts must confirm.

## gotchas--notes

### Q1
Why can a successful process exit still represent incomplete conversion?

### A1
Missing ligand/protein pairs are skipped without making the command fail, so success status alone does not prove full dataset coverage.

### Q2
What stale-state risk comes from using `exist_ok=True` on reruns?

### A2
Expected target files are overwritten, but unrelated directories left from older datasets remain in the output tree.

### Q3
Why should users not expect graph files after this command finishes?

### A3
The converter only reorganizes structure files; upstream pocket extraction and graph generation are separate later steps.
