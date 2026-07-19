# quiz: walkthrough-seminar-csv

## purpose

### Q1
Why must every pipeline component share one definition of complex ID?

### A1
Converters, structure directories, and trainers must agree with the stems of `.dgl` graph files or they will refer to different or missing complexes.

### Q2
What schema difference does this module hide from its callers?

### A2
Prototype CSVs already provide ligand and protein filenames, while full-data CSVs provide four metadata fields from which IDs and refined filenames are constructed.

### Q3
Why does the output preserve first-seen order?

### A3
It gives deterministic dataset ordering while still collapsing duplicate rows to one complex record.

## how-it-fits-in

### Q1
Why do both converters import `complex_records`?

### A1
They need the same ID, ligand filename, and protein filename interpretation when reorganizing inputs or predictions.

### Q2
Why do trainers use `complex_ids` instead of independently parsing CSV fields?

### A2
`complex_ids` projects the shared records directly to graph stems, preventing training selection from diverging from conversion.

### Q3
What relationship must hold between a returned ID and preprocessing output?

### A3
The ID must exactly equal the stem of the corresponding saved `.dgl` graph so availability filtering and dataset loading can find it.

## walkthrough

### Q1
How does `_rows_to_triples` decide which schema branch to use?

### A1
Presence of `ligand_file_name` selects the prototype filename branch; otherwise it expects all four ordered metadata columns.

### Q2
Why is pandas called with `dtype=str`?

### A2
It prevents identifier components such as residue number `210` from becoming `210.0`, which would change filenames and graph IDs.

### Q3
How do the `seen` set and `records` list serve different purposes?

### A3
The set makes duplicate detection efficient, while the list preserves the first occurrence's order and full triple.

## gotchas--notes

### Q1
Why does a partial prototype schema fail less clearly than a fully unknown schema check might?

### A1
The branch checks only for `ligand_file_name`, then accesses `protein_file_name` directly and can raise a missing-column error.

### Q2
Why is the checked-in PoseBusters CSV incompatible with the helper?

### A2
It uses `ligand_file` rather than `ligand_file_name` and does not provide the four full metadata columns expected by the fallback branch.

### Q3
What is risky about using `replace` in `_strip_suffix`?

### A3
It removes matching text wherever it occurs, not strictly at the filename end, although valid inputs are expected to use those strings only as suffixes.
