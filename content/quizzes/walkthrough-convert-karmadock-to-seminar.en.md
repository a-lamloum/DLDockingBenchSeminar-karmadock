# quiz: walkthrough-convert-karmadock-to-seminar

## purpose

### Q1
Why must repeated KarmaDock predictions be combined into one SDF per complex?

### A1
The seminar evaluator expects one ranked multi-record `<id>_pred.sdf` file for each complex rather than KarmaDock's repeat-by-repeat output tree.

### Q2
Why is MDN score used before writing the output records?

### A2
It provides the model's ranking signal, so sorting scores descending places the predicted best pose first for evaluator top-1 rank.

### Q3
What would be lost if the converter wrote only the highest-scoring molecule?

### A3
Lower-ranked candidates and best-of-N evaluation would be lost because the evaluator uses SDF record order as pose rank.

## how-it-fits-in

### Q1
Why does `run_infer.sh` invoke this converter three times?

### A1
It exports parallel raw, force-field-corrected, and alignment-corrected result trees through the same evaluator-compatible interface.

### Q2
How do `--mode` and `--n_repeat` affect the produced SDF?

### A2
`--mode` selects which pose filename variant to seek, while `--n_repeat` controls which repeat score tables and pose directories can contribute records.

### Q3
Why does the converter also need the dataset CSV after docking is complete?

### A3
The CSV supplies the ordered complex IDs and filenames needed to create one expected output path for every dataset complex.

## walkthrough

### Q1
What happens when one repeat CSV is missing versus when every repeat CSV is missing?

### A1
An individual missing repeat is ignored, but having no repeat score tables is fatal because there is no usable set of repeat outputs to convert.

### Q2
How is a complex with no score rows ranked, and why is the result deterministic?

### A2
Available repeats receive tied score `0.0`, and Python's stable descending sort preserves their repeat order.

### Q3
Why does the converter retain `KarmaDock_Score`, `KarmaDock_Repeat`, and `KarmaDock_Mode` properties?

### A3
They preserve ranking and provenance information in the exported molecular records even after the original repeat directories are no longer visible to the evaluator.

## gotchas--notes

### Q1
Why can changing write order change evaluation without changing any coordinates?

### A1
The evaluator interprets SDF file order as rank, so reordering identical records changes which pose counts as top-1.

### Q2
Why does `KarmaDock_Mode=ff_corrected` not prove that the stored coordinates were force-field corrected?

### A2
If the corrected file is missing, the converter falls back to the raw pose but records the requested mode rather than the actual fallback source.

### Q3
How can an existing pose be omitted when its repeat has no score row for that complex?

### A3
Once any scores exist for a complex, only repeats represented in those score rows enter the ranked list, even if another repeat directory contains a pose file.
