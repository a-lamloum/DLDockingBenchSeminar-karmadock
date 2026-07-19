window.KD_SEARCH_INDEX = {
  "entries": [
    {
      "excerpt": "Public KarmaDock ships inference + released weights only — there is no training script. Our contribution is everything needed to train it and to benchmark it reproducibly on the seminar split. The upstream model, preproc…",
      "heading": "1. What we did, and why (changes vs. upstream KarmaDock)",
      "text": "overview 1. what we did, and why (changes vs. upstream karmadock) public karmadock ships inference + released weights only — there is no training script. our contribution is everything needed to train it and to benchmark it reproducibly on the seminar split. the upstream model, preprocessing, and docking code are used unmodified — this keeps the benchmark faithful to the published method. file (ours) what it is why it exists scripts/train.py full checkpointed training loop (mdn + docking rmsd losses, 2-stage ) our main artifact — upstream has no trainer; required to train from scratch / fine-tune scripts/runtrain.sh preprocess + train: scratch = paper's 2-stage protocol (p2), finetune = single-stage from the released weights (p3) reproduce the paper's training protocol (methods, p.801) scripts/convertseminartokarmadock.py seminar data layout → karmadock layout the seminar's ligandrefined.sdf / proteinrefined.pdb files must be re-laid-out for karmadock's preprocessing.py scripts/convertkarmadocktoseminar.py karmadock poses → seminar results//pred.sdf (best-pose-first) produce the exact format evaluation.py expects scripts/runinfer.sh preprocess → dock → export the 3 pose variants (uncorrected / ff / align) produce the predicted poses on the cluster scripts/evaluate.sh run evaluation.py over every pipeline × variant — the separate scoring step produce the official rmsd csvs condor/.sub htcondor docker-universe submit files (3 docking jobs + 1 eval job + 2 training jobs) run everything on the sic cluster dockerfile image → ahlamloum/karmadock-seminar:v6 reproducible environment (karmadock + rdkit + torch + other dependencies backed in) what is the central contribution of this seminar project beyond upstream karmadock? why is keeping the upstream karmadock model and docking code unchanged important for the benchmark? which script is the project's main implementation artifact, and what two training modes does the surrounding workflow support?",
      "title": "Overview",
      "url": "index.html#1-what-we-did-and-why-changes-vs-upstream-karmadock"
    },
    {
      "excerpt": "① Training — produces the P2 / P3 checkpoints: Preprocess prototrain (712) into graphs, then train: --initmodel picks the route — P2 from scratch (Stage 1 MDN scoring → Stage 2 + docking RMSD) or P3 fine-tune from the re…",
      "heading": "Workflow",
      "text": "pipelines workflow 1 training — produces the p2 / p3 checkpoints: preprocess prototrain (712) into graphs, then train: --initmodel picks the route — p2 from scratch (stage 1 mdn scoring → stage 2 + docking rmsd) or p3 fine-tune from the released weights; earlystopper keeps the best epoch as the checkpoint. 2 inference & evaluation: run once per pipeline (p1 / p2 / p3): preprocess prototest (136), dock + score with the chosen weights, export the 3 pose variants (uncorrected / ff / align), then score with the official evaluation.py (symmetry-corrected rmsd, top-1). how do p1, p2, and p3 differ in where their weights come from? what determines whether the training workflow produces p2 or p3? why does inference export three pose variants before evaluation?",
      "title": "Pipelines",
      "url": "pipelines.html#workflow"
    },
    {
      "excerpt": "The tables below use the prototype prototest (136). The final full-data submission is being finalized: our from-scratch model retrained on the full seminar split (model/fullscratchkarmadockteam002.pkl; 2-stage paper prot…",
      "heading": "🚧 Final submission (full-data) — report in progress",
      "text": "results 🚧 final submission (full-data) — report in progress the tables below use the prototype prototest (136). the final full-data submission is being finalized: our from-scratch model retrained on the full seminar split (model/fullscratchkarmadockteam002.pkl; 2-stage paper protocol, stage-2 via 2×a100 (40gb each); evaluated head-to-head against the authors' released weights on the same fulltest (6,183) and posebustersfiltered (308) sets. preliminary headline (top-1 success@2 a, using evaluation.py, uncorrected): set ours (full-data) released weights fulltest (6,183) 82.2 % 88.3 % posebusters (308) 76.9 % (pb-valid 5.2 %) 83.1 % (pb-valid 2.6 %) this commit adds the trained model (model/), the evaluation input csvs (data/), the hpc condor submit files (condor/fullstage22gpu.sub = the 2×a100 stage-2 run; condor/{fulltest,posebusters}infer.sub = inference) and their job logs (condor/logs/). the full report and results notebook are being finalized. why should the align-corrected success rate not be read as the model's raw docking accuracy? what does the headline success@2a metric count? which prototype result is the primary submission result, and where is its per-complex data stored?",
      "title": "Results",
      "url": "results.html#final-submission-full-data--report-in-progress"
    },
    {
      "excerpt": "Our submission model is the from-scratch P2 (the seminar task — retrain the tool on the shared split); P1 (released baseline) and P3 (fine-tune, bonus) are shown for context. Scored by the official evaluation/evaluation.…",
      "heading": "3. Results",
      "text": "results 3. results our submission model is the from-scratch p2 (the seminar task — retrain the tool on the shared split); p1 (released baseline) and p3 (fine-tune, bonus) are shown for context. scored by the official evaluation/evaluation.py on the 136-complex prototest, for all three karmadock pose post-processing variants. success@2 a (top-1): pipeline uncorrected ff-corrected align-corrected p2 — from scratch (our model) 10.3 % 11.0 % 94.1 % p1 — baseline (released) 80.9 % 78.7 % 95.6 % p3 — fine-tune (bonus) 80.1 % 75.0 % 94.9 % uncorrected @1 a / median rmsd: p2 3.7 % / 3.38 a · p1 8.1 % / 1.45 a · p3 7.4 % / 1.48 a. per-complex csvs are in results/ (evaluation.csv, and prototestevaluation.csv = the p2 headline). numbers are deterministic (--randomseed 2023) we belive that align-corrected is misleading as according to the paper it should be lower than the uncorrected and ff-corrected results which is the opposite of what we got. [ need futher investigation - doesn't affect the training process for the next phase ] we belive that align-corrected is misleading as according to the paper it should be lower than the uncorrected and ff-corrected results which is the opposite of what we got. [ need futher investigation - doesn't affect the training process for the next phase ] the uncorrected pose is the raw model output. the ff variant is a force-field relaxation. the align-corrected variant superimposes the predicted ligand onto the reference frame.",
      "title": "Results",
      "url": "results.html#3-results"
    },
    {
      "excerpt": "Pose files are pred.sdf, conformers ranked best-MDN-first. The three variants are the KarmaDock pose post-processing options: uncorrected (raw network output), FF (force-field relaxation), align (superimpose onto the ref…",
      "heading": "Layout",
      "text": "results layout pose files are pred.sdf, conformers ranked best-mdn-first. the three variants are the karmadock pose post-processing options: uncorrected (raw network output), ff (force-field relaxation), align (superimpose onto the reference frame).",
      "title": "Results",
      "url": "results.html#layout"
    },
    {
      "excerpt": "complexid, dataset, poserank, rmsd, rmsdlt2, rmsdlt1, ligandfile, proteinfile (RDKit GetBestRMS, symmetry-corrected, top-1). success@2Å = fraction of poserank == 1 rows with rmsdlt2 == True.",
      "heading": "Eval CSV columns",
      "text": "results eval csv columns complexid, dataset, poserank, rmsd, rmsdlt2, rmsdlt1, ligandfile, proteinfile (rdkit getbestrms, symmetry-corrected, top-1). success@2a = fraction of poserank == 1 rows with rmsdlt2 == true.",
      "title": "Results",
      "url": "results.html#eval-csv-columns"
    },
    {
      "excerpt": "p1baselinealignevaluation.csv p1baselineffevaluation.csv p1baselineuncorrectedevaluation.csv p2scratchalignevaluation.csv p2scratchffevaluation.csv p2scratchuncorrectedevaluation.csv p3finetunealignevaluation.csv p3finet…",
      "heading": "Per-complex evaluation downloads",
      "text": "results per-complex evaluation downloads p1baselinealignevaluation.csv p1baselineffevaluation.csv p1baselineuncorrectedevaluation.csv p2scratchalignevaluation.csv p2scratchffevaluation.csv p2scratchuncorrectedevaluation.csv p3finetunealignevaluation.csv p3finetuneffevaluation.csv p3finetuneuncorrectedevaluation.csv prototestevaluation.csv",
      "title": "Results",
      "url": "results.html#per-complex-evaluation-downloads"
    },
    {
      "excerpt": "results/prototest/ holds the from-scratch (P2) poses — our submission's primary model. To score a different pipeline/variant, point results/prototest/ at it, e.g. rm results/prototest && ln -s p1baseline/prototest result…",
      "heading": "A. Score the shipped poses",
      "text": "reproduce a. score the shipped poses results/prototest/ holds the from-scratch (p2) poses — our submission's primary model. to score a different pipeline/variant, point results/prototest/ at it, e.g. rm results/prototest && ln -s p1baseline/prototest results/prototest then re-run. pre-computed csvs for every pipeline × variant are in results/. what is the shortest path to verify the shipped p2 poses without using the cluster? why can the htcondor submission files run without site-specific path edits? what is the difference between scoring the shipped poses and regenerating them?",
      "title": "Reproduce",
      "url": "reproduce.html#a-score-the-shipped-poses"
    },
    {
      "excerpt": "The subs (condor/) are portable (they transfer the data + our code into the job and use the released weights baked in the image — no path edits). Docking is deterministic (--randomseed 2023) The evaluation set is data/pr…",
      "heading": "B. Regenerate the poses on the cluster (optional)",
      "text": "reproduce b. regenerate the poses on the cluster (optional) the subs (condor/) are portable (they transfer the data + our code into the job and use the released weights baked in the image — no path edits). docking is deterministic (--randomseed 2023) the evaluation set is data/prototest.csv (136 complexes). the reference structures come from the bundle. prototrain (712) is unchanged. the evaluation set is data/prototest.csv (136 complexes). the reference structures come from the bundle. prototrain (712) is unchanged.",
      "title": "Reproduce",
      "url": "reproduce.html#b-regenerate-the-poses-on-the-cluster-optional"
    },
    {
      "excerpt": "Hyper-parameters are in §5; the cluster drivers are condor/p2trainscratch.sub (P2) and condor/p3finetune.sub (P3). Training is the long path (on one GPU ~54 hours for the 712 complexes in prototrain).",
      "heading": "Retraining from scratch and fine-tuning",
      "text": "reproduce retraining from scratch and fine-tuning hyper-parameters are in §5; the cluster drivers are condor/p2trainscratch.sub (p2) and condor/p3finetune.sub (p3). training is the long path (on one gpu ~54 hours for the 712 complexes in prototrain).",
      "title": "Reproduce",
      "url": "reproduce.html#retraining-from-scratch-and-fine-tuning"
    },
    {
      "excerpt": "Computed directly from the checked-in per-epoch logs. Best values are minima. run epochs best val loss best val RMSD final val loss logged time CSV Full-data Stage 2 470 3.55414 (epoch 399) 2.81194 (epoch 399) 3.62990 81…",
      "heading": "Training-log summary",
      "text": "training training-log summary computed directly from the checked-in per-epoch logs. best values are minima. run epochs best val loss best val rmsd final val loss logged time csv full-data stage 2 470 3.55414 (epoch 399) 2.81194 (epoch 399) 3.62990 81.50 h fullstage2trainlog.csv p2 stage 1 370 1.20531 (epoch 299) n/a (posr=0) 1.21701 1.95 h p2stage1trainlog.csv p2 stage 2 146 6.16796 (epoch 125) 4.94560 (epoch 125) 6.24452 3.42 h p2stage2trainlog.csv p3 fine-tune 68 3.48925 (epoch 37) 2.58530 (epoch 37) 3.59316 1.57 h p3finetunetrainlog.csv why is posr set to 0 in p2 stage 1 and to 1 in stage 2? how is the effective batch size of 64 obtained? why is p2 stage 1 validation rmsd reported as unavailable even though the log contains zeros?",
      "title": "Training",
      "url": "training.html#training-log-summary"
    },
    {
      "excerpt": "path description README.md this file Dockerfile image (ahlamloum/karmadock-seminar:v6) scripts/ train.py (main artifact), converters, runinfer.sh, evaluate.sh, runtrain.sh condor/ portable HTCondor submit files (3 dockin…",
      "heading": "6. Repository layout",
      "text": "reference 6. repository layout path description readme.md this file dockerfile image (ahlamloum/karmadock-seminar:v6) scripts/ train.py (main artifact), converters, runinfer.sh, evaluate.sh, runtrain.sh condor/ portable htcondor submit files (3 docking + 1 eval + 2 training) evaluation/evaluation.py the seminar's official evaluator (unmodified) model/ p2 + p3 trained checkpoints (~15 mb each) notebooks/resultsandcomparison.ipynb tables, charts results/ prototest/ = p2 poses that evaluation.py --dataset prototest scores; /prototest{,ff,align}/ = all 3 pipelines × 3 variants; evaluation.csv = official rmsd per pipeline × variant docs/ training logs + figures data/prototest.csv prototest mapping (136 complexes) data/prototypemodeldata.zip reference structures: prototest + prototrain, refined sdf/pdb scripts/readme.md provenance: our code vs. upstream karmadock docker image: ahlamloum/karmadock-seminar:v6 (docker hub). which parts belong to upstream karmadock, and which parts are the seminar team's work? why was reducing the requested gpu count a practical fix on the cluster? what evidence in the repository helps distinguish authored code from reused upstream code?",
      "title": "Reference",
      "url": "reference.html#6-repository-layout"
    },
    {
      "excerpt": "Repository root README.md Dockerfile .gitignore scripts/ condor/ condor/logs/ data/ docs/ evaluation/ model/ notebooks/ results/…",
      "heading": "File-by-file guide",
      "text": "reference file-by-file guide repository root readme.md dockerfile .gitignore scripts/ condor/ condor/logs/ data/ docs/ evaluation/ model/ notebooks/ results/ plain-english explanations for every meaningful file in the karmadock team-002 submission",
      "title": "Reference",
      "url": "reference.html#file-by-file-guide"
    },
    {
      "excerpt": "README.md is the main write-up. Dockerfile builds ahlamloum/karmadock-seminar:v6 from the authors' pre-packed environment. .gitignore keeps the repo portable.",
      "heading": "Repository root",
      "text": "reference file-by-file guide repository root readme.md main write-up three pipelines results evaluation reproduction training parameters layout known issues dockerfile reproducible image ahlamloum/karmadock-seminar:v6 miniconda upstream karmadock pre-packed conda environment conda-pack zenodo karmadock_env.yaml dependency solver prefetch_generator rmsd wandb scripts/ pythonpath=/app/karmadock import check .gitignore prototype data __pycache__ notebook checkpoints data/ symlinks *.csv split files",
      "title": "Reference",
      "url": "reference.html#guide-repository-root"
    },
    {
      "excerpt": "The team's training loop, training and inference drivers, data-layout converters, CSV helper, and provenance note, with deep-dive pages linked.",
      "heading": "scripts/ (the team's code — deep-dive pages linked)",
      "text": "reference file-by-file guide scripts/ scripts/train.py two-stage training mdn rmsd optimizer early stopping gradient accumulation deterministic train/val split weights & biases --resume scripts/run_train.sh scratch p2 finetune p3 proto_train 712 scripts/run_full_train.sh full_train 23,483 full_val 2,609 --val_csv --val_graph_dir scripts/run_infer.sh preprocess dock uncorrected ff-relaxed align-corrected scripts/evaluate.sh evaluation.py pipeline pose variant rmsd csv scripts/convert_seminar_to_karmadock.py ligand_refined.sdf protein_refined.pdb scripts/convert_karmadock_to_seminar.py results/<ds>/<id>_pred.sdf best-pose-first scripts/seminar_csv.py csv schema mappings scripts/readme.md provenance no upstream source modified",
      "title": "Reference",
      "url": "reference.html#guide-scripts"
    },
    {
      "excerpt": "HTCondor submit files for P1/P2/P3 inference and evaluation, prototype and full-data training, full_test inference, and PoseBusters inference.",
      "heading": "condor/ (HTCondor submit files — deep-dive linked)",
      "text": "reference file-by-file guide condor/ htcondor condor/p1_baseline.sub p1 released weights evaluation condor/p2_scratch_infer.sub p2 from-scratch checkpoint condor/p3_finetune_infer.sub p3 fine-tuned checkpoint condor/p2_train_scratch.sub two-stage from scratch condor/p3_finetune.sub fine-tune training condor/evaluate.sub evaluate.sh all pipelines variants condor/full_stage2_2gpu.sub full-data stage-2 2xa100 distributed hpc cluster-side scripts condor/full_train_scratch.sub full-data from-scratch condor/full_test_infer.sub full_test 6,183 condor/posebusters_infer.sub posebusters 308",
      "title": "Reference",
      "url": "reference.html#guide-condor"
    },
    {
      "excerpt": "HPC job logs that document the completed 2xA100 Stage-2 run and the head-to-head full_test and PoseBusters inference comparisons.",
      "heading": "condor/logs/ (HPC job logs — provenance for the reported numbers)",
      "text": "reference file-by-file guide condor/logs/ hpc logs provenance condor/logs/s2_2gpu.169253.* stdout stderr log completed 2xa100 full-data stage-2 exit 0 condor/logs/full_test_infer.169842.* full_test_released.169868.* full_test team model authors released weights head-to-head condor/logs/posebusters_infer.169843.* posebusters_released.169869.* posebusters head-to-head",
      "title": "Reference",
      "url": "reference.html#guide-condor-logs"
    },
    {
      "excerpt": "Evaluation-set CSV mappings for proto_test, full_test, and PoseBusters, plus the reference crystal-structure bundle.",
      "heading": "data/ (evaluation-set mappings + reference structures)",
      "text": "reference file-by-file guide data/ data/proto_test.csv 136 ligand protein file names year log binding affinity measurement type pdb id data/full_test.csv 6,183 same columns data/posebusters_filtered.csv 308 ligand name ligand file protein file data/prototype_model_data.zip reference crystal structures proto_test proto_train refined sdf pdb local scoring",
      "title": "Reference",
      "url": "reference.html#guide-data"
    },
    {
      "excerpt": "Per-epoch P2 Stage-1, P2 Stage-2, and P3 fine-tune curves, plus the training and inference workflow diagrams.",
      "heading": "docs/ (training curves + figures)",
      "text": "reference file-by-file guide docs/ docs/p2_stage1_train_log.csv per-epoch p2 stage-1 mdn scoring epoch train/val loss rmsd mdn seconds docs/p2_stage2_train_log.csv p2 stage-2 docking scoring docs/p3_finetune_train_log.csv p3 fine-tune docs/workflow_training.png docs/workflow_inference.png training inference workflow diagrams",
      "title": "Reference",
      "url": "reference.html#guide-docs"
    },
    {
      "excerpt": "The seminar's official, unmodified evaluator for symmetry-corrected top-1 pose RMSD and PoseBusters validity checks.",
      "heading": "evaluation/",
      "text": "reference file-by-file guide evaluation/ evaluation/evaluation.py official unmodified evaluator symmetry-corrected top-1 pose rmsd posebusters validity checks",
      "title": "Reference",
      "url": "reference.html#guide-evaluation"
    },
    {
      "excerpt": "The P2 prototype checkpoint, P3 fine-tune checkpoint, and final full-data from-scratch checkpoint, about 15 MB each.",
      "heading": "model/ (trained checkpoints, ~15 MB each)",
      "text": "reference file-by-file guide model/ trained checkpoints 15 mb model/p2_scratch_karmadock_team002.pkl p2 from-scratch prototype 712 model/p3_finetune_karmadock_team002.pkl p3 fine-tune bonus model/full_scratch_karmadock_team002.pkl full-data final submission two-stage stage-2 2xa100 headline",
      "title": "Reference",
      "url": "reference.html#guide-model"
    },
    {
      "excerpt": "The analysis notebook containing results tables and charts that compare pipelines and pose variants.",
      "heading": "notebooks/",
      "text": "reference file-by-file guide notebooks/ notebooks/results_and_comparison.ipynb analysis notebook results tables charts pipelines pose variants",
      "title": "Reference",
      "url": "reference.html#guide-notebooks"
    },
    {
      "excerpt": "The primary P2 poses, every pipeline x pose-variant result directory, and the official per-complex RMSD CSVs, shipped unzipped.",
      "heading": "results/ (predicted poses + scored CSVs — shipped unzipped)",
      "text": "reference file-by-file guide results/ results/proto_test/ p2 from-scratch predicted poses evaluation.py --dataset proto_test primary result set results/{p1_baseline,p2_scratch,p3_finetune}/proto_test{,_ff,_align}/ three pipelines three variants uncorrected ff-relaxed align-corrected 136 results/*_evaluation.csv official per-complex rmsd csv pipeline variant",
      "title": "Reference",
      "url": "reference.html#guide-results"
    },
    {
      "excerpt": "Node Problem: in the idun cluster the gpu nodes are sometime during the run just return error so we had to exclude this specific node as a requirement in our sub files. Fix: exclude it — requirements = … && (Machine =!=…",
      "heading": "7. Issues & fixes",
      "text": "reference 7. issues & fixes node problem: in the idun cluster the gpu nodes are sometime during the run just return error so we had to exclude this specific node as a requirement in our sub files. fix: exclude it — requirements = ... && (machine =!= \"idun.hpc.uni-saarland.de\"). [solved]. resources limitations: requestgpus=2/4 jobs sat idle for days. the prototype runs single-gpu. we couldn't run on 2 gpus that's why we changed it to 1 gpu, we are not sure if this gonna work on the full dataset as it will take too much time. fix: change the request to 1 gpu [solved].",
      "title": "Reference",
      "url": "reference.html#7-issues--fixes"
    },
    {
      "excerpt": "The key point: public KarmaDock (schrojunzhang/KarmaDock) ships inference code + pretrained weights only — there is NO training script. Our seminar contribution is the training loop, the data adapters, the run wrappers,…",
      "heading": "Authorship & provenance",
      "text": "reference authorship & provenance the key point: public karmadock (schrojunzhang/karmadock) ships inference code + pretrained weights only — there is no training script. our seminar contribution is the training loop, the data adapters, the run wrappers, and the whole htcondor orchestration built around karmadock. we call karmadock's modules as-is (model, preprocessing, docking) and did not modify any file inside the upstream karmadock/ — it is cloned fresh in the dockerfile; all of our code lives in scripts/.",
      "title": "Reference",
      "url": "reference.html#authorship--provenance"
    },
    {
      "excerpt": "Files we created",
      "heading": "Files we created",
      "text": "reference files we created",
      "title": "Reference",
      "url": "reference.html#files-we-created"
    },
    {
      "excerpt": "file what it does KarmaDock pieces it calls train.py our main artifact — a complete checkpointed training/fine-tuning loop (KarmaDock has none): loop, optimizer, early-stopping, gradient accumulation, val split, W&B, --r…",
      "heading": "scripts/",
      "text": "reference scripts/ file what it does karmadock pieces it calls train.py our main artifact — a complete checkpointed training/fine-tuning loop (karmadock has none): loop, optimizer, early-stopping, gradient accumulation, val split, w&b, --resume. imports the upstream karmadock model, pdbbindgraphdataset, passnonedataloader, setrandomseed/earlystopper; uses the model's own forward() losses convertseminartokarmadock.py seminar data layout → karmadock layout — convertkarmadocktoseminar.py karmadock docked poses → seminar pred.sdf (best-first) reads pose sdfs (rdkit) runinfer.sh portable docking: preprocess → dock → export the 3 pose variants (uncorrected/ff/align) calls upstream preprocessing.py, generategraph.py, liganddocking.py + our converters evaluate.sh the separate scoring step: runs evaluation.py over every pipeline × variant calls the seminar's evaluation/evaluation.py runtrain.sh portable training: scratch = paper 2-stage (p2), finetune = from released weights (p3); preprocesses prototrain first calls our train.py",
      "title": "Reference",
      "url": "reference.html#scripts"
    },
    {
      "excerpt": "file job p1baseline.sub P1 inference (released weights) + eval p2scratchinfer.sub P2 inference (our from-scratch checkpoint) + eval p3finetuneinfer.sub P3 inference (our fine-tuned checkpoint) + eval p2trainscratch.sub P…",
      "heading": "condor/ (our work)",
      "text": "reference condor/ (our work) file job p1baseline.sub p1 inference (released weights) + eval p2scratchinfer.sub p2 inference (our from-scratch checkpoint) + eval p3finetuneinfer.sub p3 inference (our fine-tuned checkpoint) + eval p2trainscratch.sub p2 training (2-stage from scratch) p3finetune.sub p3 fine-tune training evaluate.sub run the evaluate.sh script",
      "title": "Reference",
      "url": "reference.html#condor-our-work"
    },
    {
      "excerpt": "Dockerfile — builds ahlamloum/karmadock-seminar:v6 (clones KarmaDock, installs the authors' packed conda env, adds our scripts/). model/ checkpoints (P2, P3), the results notebook, all docs, and the training logs in docs…",
      "heading": "Other ours",
      "text": "reference other ours dockerfile — builds ahlamloum/karmadock-seminar:v6 (clones karmadock, installs the authors' packed conda env, adds our scripts/). model/ checkpoints (p2, p3), the results notebook, all docs, and the training logs in docs/.",
      "title": "Reference",
      "url": "reference.html#other-ours"
    },
    {
      "excerpt": "Authors': the KarmaDock model + preprocessing/docking utilities + released weights (used as-is). Ours: train.py (there was no trainer), the data adapters, the run wrappers, every condor .sub, the Docker build, and all an…",
      "heading": "Summary",
      "text": "reference summary authors': the karmadock model + preprocessing/docking utilities + released weights (used as-is). ours: train.py (there was no trainer), the data adapters, the run wrappers, every condor .sub, the docker build, and all analysis/docs. no karmadock source file was edited.",
      "title": "Reference",
      "url": "reference.html#summary"
    },
    {
      "excerpt": "The end-to-end flow is: Convert the input layout. seminarcsv.py normalizes the supported CSV schemas. convertseminartokarmadock.py copies each ligand/protein pair into upstream KarmaDock's naming convention. Preprocess a…",
      "heading": "How these files fit together",
      "text": "code walkthroughs how these files fit together the end-to-end flow is: convert the input layout. seminarcsv.py normalizes the supported csv schemas. convertseminartokarmadock.py copies each ligand/protein pair into upstream karmadock's naming convention. preprocess and train. shell drivers call upstream preprocessing.py and generategraph.py, then this repository's train.py for single-process training. the full-data stage-2 run used the repository's condor job configuration to launch one process per gpu. training can start randomly with the two-stage scoring-then-docking protocol or fine-tune released weights. infer. runinfer.sh preprocesses a test set and calls upstream liganddocking.py to produce raw, force-field-minimized, and aligned poses. convert predictions back. convertkarmadocktoseminar.py ranks repeats by mdn score and writes pred.sdf files in evaluator order. evaluate. evaluation.py computes symmetry-corrected heavy-atom rmsd and optional posebusters checks. evaluate.sh applies it to all prototype pipeline/pose variants. the run.sh files are reproducible orchestration layers around those python programs. the files in condor/ sit above the shell wrappers: they select the docker image, stage inputs/outputs, reserve resources, pass arguments/environment variables, and route logs. why does the pipeline need one converter before karmadock and another converter after docking? how do the single-process trainer and the full-data condor job support different training roles without changing the upstream karmadock model? why are the files in condor/ described as sitting above the run.sh wrappers?",
      "title": "Code walkthroughs",
      "url": "walkthroughs.html#how-these-files-fit-together"
    },
    {
      "excerpt": "Source file Walkthrough scripts/train.py Single-process training scripts/seminarcsv.py CSV-schema normalization scripts/convertseminartokarmadock.py Input-layout conversion script…",
      "heading": "Walkthrough index",
      "text": "code walkthroughs walkthrough index source file walkthrough scripts/train.py single-process training scripts/seminarcsv.py csv-schema normalization scripts/convertseminartokarmadock.py input-layout conversion scripts/convertkarmadocktoseminar.py prediction-layout conversion evaluation/evaluation.py rmsd and posebusters evaluation scripts/runtrain.sh prototype training driver scripts/runfulltrain.sh full-data training driver scripts/runinfer.sh inference driver scripts/evaluate.sh batch evaluation driver all 12 condor/.sub files htcondor submit files if a student wants to understand why graph stems must match csv-derived complex ids, which walkthrough should they read first? which pair of walkthroughs best explains the transition from predicted karmadock poses to benchmark csv metrics? why does the index separate shell-driver walkthroughs from the python training and evaluation walkthroughs?",
      "title": "Code walkthroughs",
      "url": "walkthroughs.html#walkthrough-index"
    },
    {
      "excerpt": "These 12 files describe repository jobs to HTCondor: prototype/full training, single- and multi-GPU Stage-2 experiments, inference, and evaluation. They do not implement ML logic themselves; they select a Docker environm…",
      "heading": "Purpose",
      "text": "condor/.sub purpose these 12 files describe repository jobs to htcondor: prototype/full training, single- and multi-gpu stage-2 experiments, inference, and evaluation. they do not implement ml logic themselves; they select a docker environment, stage files, reserve resources, invoke a shell wrapper with arguments/environment, and route outputs and logs. why are the submit files not considered implementations of the machine-learning algorithms? what problem does keeping job resources and file staging in the submit files solve? why are there separate submit files for prototype, full-data, inference, evaluation, and stage-2 experiments?",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#purpose"
    },
    {
      "excerpt": "Users submit one with condorsubmit .sub. HTCondor matches it to a worker, transfers declared inputs into a temporary job sandbox, runs the declared shell executable inside ahlamloum/karmadock-seminar:v6, and transfers de…",
      "heading": "How it fits in",
      "text": "condor/.sub how it fits in users submit one with condorsubmit .sub. htcondor matches it to a worker, transfers declared inputs into a temporary job sandbox, runs the declared shell executable inside ahlamloum/karmadock-seminar:v6, and transfers declared outputs back. inference jobs call scripts/runinfer.sh, training jobs call one of the training wrappers, ddp jobs call a stage-2 launcher, and the cpu evaluation job calls scripts/evaluate.sh after predictions exist. what happens between condorsubmit and execution of a repository wrapper? why must condor/evaluate.sub run after the inference jobs? how do the inference, training, ddp, and evaluation submit files connect to the lower orchestration layer?",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#how-it-fits-in"
    },
    {
      "excerpt": "Walkthrough",
      "heading": "Walkthrough",
      "text": "condor/.sub walkthrough why does queue 1 not imply that a ddp job uses only one gpu process? why do full-data and ddp jobs rely on a mounted home instead of transferring all inputs and checkpoints? how do the two-gpu final job and the smoke job use resource settings for different goals?",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#walkthrough"
    },
    {
      "excerpt": "A submit file is a list of key = value directives plus queue. The common directives here mean: universe = docker runs the executable inside dockerimage. The image supplies KarmaDock, Python, PyTorch, RDKit, and other dep…",
      "heading": "Shared HTCondor submit-file vocabulary",
      "text": "condor/.sub shared htcondor submit-file vocabulary a submit file is a list of key = value directives plus queue. the common directives here mean: universe = docker runs the executable inside dockerimage. the image supplies karmadock, python, pytorch, rdkit, and other dependencies. executable is the staged program; arguments supplies its positional command line. environment injects job-specific environment variables. requestgpus, requestcpus, and requestmemory advertise required resources to matching/scheduling and tell the worker how much to allocate. these are per job, not per process. requirements is an htcondor classad expression. every file requires saarland's cs uid domain and excludes idun.hpc.uni-saarland.de, a node documented in the repository as problematic. +wantgpuhomemounted = true is a site-specific custom attribute requesting the submitter's home inside the container. gpusminimumcapability and gpusminimummemory further constrain gpu matching. compute capability 8.0 or newer and at least 32,000 mb are requested only by the final 2-gpu job. shouldtransferfiles = yes enables file staging. transferinputfiles lists code/data copied into the sandbox; directories are transferred recursively. whentotransferoutput = onexit waits until normal job termination, and transferoutputfiles restricts what is copied back. jobs that persist results directly in a mounted home do not need large output transfers. output captures program standard output, error captures standard error, and log is htcondor's lifecycle/event log. $(clusterid) and $(procid) prevent jobs from sharing log filenames. queue 1 materializes one process from the description. this is unrelated to ddp process count: a single condor job can start multiple local gpu processes with torchrun.",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#shared-htcondor-submit-file-vocabulary"
    },
    {
      "excerpt": "This is the sole CPU-only job: there is no requestgpus. It stages all prototype results in, runs the 3×3 RMSD matrix, and returns the augmented results tree. Four CPUs and 16 GB serve RDKit/evaluation rather than model i…",
      "heading": "condor/evaluate.sub",
      "text": "condor/.sub condor/evaluate.sub this is the sole cpu-only job: there is no requestgpus. it stages all prototype results in, runs the 3×3 rmsd matrix, and returns the augmented results tree. four cpus and 16 gb serve rdkit/evaluation rather than model inference. it must run after inference because its entire input is the poses those jobs produce.",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#evaluate-sub"
    },
    {
      "excerpt": "This is the concrete final two-GPU experiment. Environment overrides maintain global effective batch 64 and point directly to the final Stage-1 best checkpoint. The v2 launcher changes Stage-2 weight decay/patience. GPU…",
      "heading": "condor/fullstage22gpu.sub",
      "text": "condor/.sub condor/fullstage22gpu.sub this is the concrete final two-gpu experiment. environment overrides maintain global effective batch 64 and point directly to the final stage-1 best checkpoint. the v2 launcher changes stage-2 weight decay/patience. gpu constraints avoid known 16 gb out-of-memory devices. code alone is transferred because data, graphs, checkpoint, and outputs live on the mounted home. the hard-coded /home/bdldtteam002/... makes this submit file account-specific despite the launcher itself having overrideable paths.",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#full-stage2-2gpu-sub"
    },
    {
      "excerpt": "This is the original four-GPU Stage-2 test using runfullstage2ddp.sh defaults: batch 8 × accumulation 2 × four ranks = 64. It requests more CPUs/memory but imposes no GPU generation/memory constraint, unlike the 2-GPU co…",
      "heading": "condor/fullstage2mgpu.sub",
      "text": "condor/.sub condor/fullstage2mgpu.sub this is the original four-gpu stage-2 test using runfullstage2ddp.sh defaults: batch 8 × accumulation 2 × four ranks = 64. it requests more cpus/memory but imposes no gpu generation/memory constraint, unlike the 2-gpu copy. it expects submission from a deployment directory containing the three bare transferred filenames and a pre-created stage-1 snapshot in the mounted home.",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#full-stage2-mgpu-sub"
    },
    {
      "excerpt": "This stages the 6,183-complex full test set and the repository's full-data checkpoint, then tags output fullscratch. The checkpoint is passed as a basename because HTCondor stages it into the sandbox; runinfer.sh resolve…",
      "heading": "condor/fulltestinfer.sub",
      "text": "condor/.sub condor/fulltestinfer.sub this stages the 6,183-complex full test set and the repository's full-data checkpoint, then tags output fullscratch. the checkpoint is passed as a basename because htcondor stages it into the sandbox; runinfer.sh resolves that basename against $pwd. it is one-gpu inference, not ddp.",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#full-test-infer-sub"
    },
    {
      "excerpt": "This calls the single-GPU full-data wrapper in scratch mode, which performs both stages. Only code is staged; the roughly 14 GB data, generated graphs, and resumable checkpoints remain on the mounted home. It asks for su…",
      "heading": "condor/fulltrainscratch.sub",
      "text": "condor/.sub condor/fulltrainscratch.sub this calls the single-gpu full-data wrapper in scratch mode, which performs both stages. only code is staged; the roughly 14 gb data, generated graphs, and resumable checkpoints remain on the mounted home. it asks for substantially more host memory than inference because graph preprocessing/training loaders are heavier.",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#full-train-scratch-sub"
    },
    {
      "excerpt": "P1 is inference-only with the absolute released checkpoint baked into the Docker image, so no model file is transferred. Its structures, CSV, scripts, and resulting results directory otherwise mirror P2/P3 for a controll…",
      "heading": "condor/p1baseline.sub",
      "text": "condor/.sub condor/p1baseline.sub p1 is inference-only with the absolute released checkpoint baked into the docker image, so no model file is transferred. its structures, csv, scripts, and resulting results directory otherwise mirror p2/p3 for a controlled comparison.",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#p1-baseline-sub"
    },
    {
      "excerpt": "P2 differs from P1 only in checkpoint source/tag: it stages the team's prototype from-scratch model and resolves its basename. Holding data, image, random seed (inside the wrapper), and resources constant isolates the ef…",
      "heading": "condor/p2scratchinfer.sub",
      "text": "condor/.sub condor/p2scratchinfer.sub p2 differs from p1 only in checkpoint source/tag: it stages the team's prototype from-scratch model and resolves its basename. holding data, image, random seed (inside the wrapper), and resources constant isolates the effect of weights.",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#p2-scratch-infer-sub"
    },
    {
      "excerpt": "This stages the compact prototype training set and requests scratch, causing MDN-only Stage 1 followed by combined Stage 2. Only ckpt is returned. The wrapper also creates worktrain, but that directory is not listed as a…",
      "heading": "condor/p2trainscratch.sub",
      "text": "condor/.sub condor/p2trainscratch.sub this stages the compact prototype training set and requests scratch, causing mdn-only stage 1 followed by combined stage 2. only ckpt is returned. the wrapper also creates worktrain, but that directory is not listed as an output, so it is useful within a rescheduled sandbox only if the site's rescheduling preserves that working state; ordinary completion transfers no graphs back.",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#p2-train-scratch-sub"
    },
    {
      "excerpt": "This is resource/data-identical to P2 training but passes finetune, so the wrapper initializes from released image weights and runs only the combined-loss stage. That symmetry makes P2/P3 training differences attributabl…",
      "heading": "condor/p3finetune.sub",
      "text": "condor/.sub condor/p3finetune.sub this is resource/data-identical to p2 training but passes finetune, so the wrapper initializes from released image weights and runs only the combined-loss stage. that symmetry makes p2/p3 training differences attributable to protocol rather than scheduling resources.",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#p3-finetune-sub"
    },
    {
      "excerpt": "P3 inference is structurally identical to P2 inference but stages the fine-tuned checkpoint and selects the p3finetune result tag.",
      "heading": "condor/p3finetuneinfer.sub",
      "text": "condor/.sub condor/p3finetuneinfer.sub p3 inference is structurally identical to p2 inference but stages the fine-tuned checkpoint and selects the p3finetune result tag.",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#p3-finetune-infer-sub"
    },
    {
      "excerpt": "This reuses the full-data model on the smaller PoseBusters benchmark. The submit file is internally clear, but the called converter currently does not recognize the checked-in PoseBusters CSV's ligandfile schema; see Got…",
      "heading": "condor/posebustersinfer.sub",
      "text": "condor/.sub condor/posebustersinfer.sub this reuses the full-data model on the smaller posebusters benchmark. the submit file is internally clear, but the called converter currently does not recognize the checked-in posebusters csv's ligandfile schema; see gotchas.",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#posebusters-infer-sub"
    },
    {
      "excerpt": "The smoke test caps training at three epochs to validate process launch, NCCL communication, ID sharding, metric reductions, and rank-0 checkpointing before spending days on a full run. Because it otherwise uses original…",
      "heading": "condor/smoke2gpu.sub",
      "text": "condor/.sub condor/smoke2gpu.sub the smoke test caps training at three epochs to validate process launch, nccl communication, id sharding, metric reductions, and rank-0 checkpointing before spending days on a full run. because it otherwise uses original launcher defaults (bs=8, acc=2), its two-gpu effective batch is 32, not 64; its purpose is systems validation rather than a paper-faithful training result.",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#smoke-2gpu-sub"
    },
    {
      "excerpt": "Submit paths are interpreted relative to the directory where condorsubmit is run, not necessarily the .sub file's directory. Prototype files expect repository-root submission; the DDP files with bare names expect a deplo…",
      "heading": "Gotchas / notes",
      "text": "condor/.sub gotchas / notes submit paths are interpreted relative to the directory where condorsubmit is run, not necessarily the .sub file's directory. prototype files expect repository-root submission; the ddp files with bare names expect a deployment directory containing those files. the logs/ parent directory generally needs to exist on the submit side; these files do not create it. onexit does not continuously copy sandbox checkpoints back. full-data/ddp jobs avoid that risk by writing into the mounted home, while prototype jobs depend on sandbox/reschedule behavior until exit. only fullstage22gpu.sub constrains gpu capability/memory. other gpu jobs can match any site gpu satisfying the generic request, including cards with different memory/performance. posebustersinfer.sub stages data/posebustersfiltered.csv, whose actual columns are ligandname,ligandfile,proteinfile. runinfer.sh calls seminarcsv.complexrecords, which does not support that schema and instead looks for full metadata columns. as checked in, this route can fail with a pandas keyerror; the submit-file comment claiming ligandfile is read is inconsistent with the code. the abbreviated # ... comments unchanged ... lines above only elide prose comments; every executable directive from all 12 source files is reproduced. why can submitting from the wrong directory break an otherwise valid .sub file? why is onexit insufficient protection for prototype checkpoints during a long interrupted job? why can posebustersinfer.sub fail before docking even though its staging directives are coherent?",
      "title": "condor/.sub",
      "url": "walkthrough-condor.html#gotchas--notes"
    },
    {
      "excerpt": "This is the repository's dataset-level docking evaluator. It maps CSV rows to reference and predicted structures, computes symmetry-aware heavy-atom RMSD for ranked poses, optionally runs PoseBusters physical-validity ch…",
      "heading": "Purpose",
      "text": "evaluation/evaluation.py purpose this is the repository's dataset-level docking evaluator. it maps csv rows to reference and predicted structures, computes symmetry-aware heavy-atom rmsd for ranked poses, optionally runs posebusters physical-validity checks, prints aggregate top-1/best-of-n statistics, and writes one result row per evaluated pose. why does this evaluator produce one row per pose rather than only one summary row per complex? what two different aspects of docking quality do rmsd and posebusters evaluate? why is symmetry awareness important when comparing ligand coordinates?",
      "title": "evaluation/evaluation.py",
      "url": "walkthrough-evaluation.html#purpose"
    },
    {
      "excerpt": "It consumes data/.csv, reference files under data//, and predictions under results//pred.sdf. scripts/evaluate.sh invokes it repeatedly through temporary symlinks for prototype comparisons; users can also invoke it direc…",
      "heading": "How it fits in",
      "text": "evaluation/evaluation.py how it fits in it consumes data/.csv, reference files under data//, and predictions under results//pred.sdf. scripts/evaluate.sh invokes it repeatedly through temporary symlinks for prototype comparisons; users can also invoke it directly. its cli requires --dataset, accepts --topn, --nopbvalid, and --outputcsv, and defaults output to results/evaluation.csv. how does the dataset name determine the evaluator's input locations? why can scripts/evaluate.sh reuse this fixed directory interface for several prototype pipelines? what changes when --nopbvalid is supplied?",
      "title": "evaluation/evaluation.py",
      "url": "walkthrough-evaluation.html#how-it-fits-in"
    },
    {
      "excerpt": "Walkthrough",
      "heading": "Walkthrough",
      "text": "evaluation/evaluation.py walkthrough why can getbestrms report an optimistic docking result for a ligand placed incorrectly in the protein pocket? what happens to reported rank when an early sdf record cannot be parsed? why are global loss-style percentages derived from sums and result rows rather than treating missing complexes as failures? how does best-of-n differ from the model's own top-1 ranking?",
      "title": "evaluation/evaluation.py",
      "url": "walkthrough-evaluation.html#walkthrough"
    },
    {
      "excerpt": "Path gives composable filesystem paths; the type hints describe expected dictionaries, lists, optional molecules/paths, and triples but do not enforce types at runtime. RDKit parses molecules and performs atom alignment.…",
      "heading": "Module setup and logging",
      "text": "evaluation/evaluation.py module setup and logging path gives composable filesystem paths; the type hints describe expected dictionaries, lists, optional molecules/paths, and triples but do not enforce types at runtime. rdkit parses molecules and performs atom alignment. a module logger provides timestamps and severity levels, useful in long cluster evaluations.",
      "title": "evaluation/evaluation.py",
      "url": "walkthrough-evaluation.html#module-setup-and-logging"
    },
    {
      "excerpt": "RMSD is the square root of the mean squared distance between corresponding atoms. Hydrogens are removed because their positions/counts are variable and docking benchmarks conventionally emphasize heavy (non-hydrogen) ato…",
      "heading": "Symmetry-corrected RMSD",
      "text": "evaluation/evaluation.py symmetry-corrected rmsd rmsd is the square root of the mean squared distance between corresponding atoms. hydrogens are removed because their positions/counts are variable and docking benchmarks conventionally emphasize heavy (non-hydrogen) atoms. symmetry correction considers chemically equivalent atom mappings—for example, swapping indistinguishable atoms in a symmetric group should not make an otherwise identical pose look wrong. getbestrms selects the mapping and rigid alignment with minimum rmsd. any mismatch or rdkit failure becomes nan so one molecule need not terminate the dataset. an important interpretation detail is that getbestrms performs optimal spatial alignment of the probe to the reference. that removes global translation/rotation before scoring, whereas pocket-conditioned docking rmsd is often measured in the fixed protein frame. this implementation therefore measures shape/conformation agreement after superposition, not purely placement error in the pocket.",
      "title": "evaluation/evaluation.py",
      "url": "walkthrough-evaluation.html#symmetry-corrected-rmsd"
    },
    {
      "excerpt": "An SDF can contain several molecule records; this project treats their file order as rank. RDKit suppliers parse lazily. sanitize=False avoids rejecting predicted geometries because of chemistry checks, allowing RMSD att…",
      "heading": "Loading predicted and reference molecules",
      "text": "evaluation/evaluation.py loading predicted and reference molecules an sdf can contain several molecule records; this project treats their file order as rank. rdkit suppliers parse lazily. sanitize=false avoids rejecting predicted geometries because of chemistry checks, allowing rmsd attempts on imperfect outputs; removehs=false preserves the record initially, with hydrogens removed explicitly during rmsd. invalid records are skipped, so returned-list position is compacted even though warning rank reports original record position. only the first parseable reference record is used. returning none gives the caller one clear missing/unparseable signal.",
      "title": "evaluation/evaluation.py",
      "url": "walkthrough-evaluation.html#loading-predicted-and-reference-molecules"
    },
    {
      "excerpt": "The complex ID links CSV rows to pred.sdf. Refined seminar datasets and PoseBusters use different suffixes. The fallback is defensive, although argparse currently restricts calls to known dataset names. replace removes a…",
      "heading": "Dataset schemas and paths",
      "text": "evaluation/evaluation.py dataset schemas and paths the complex id links csv rows to pred.sdf. refined seminar datasets and posebusters use different suffixes. the fallback is defensive, although argparse currently restricts calls to known dataset names. replace removes all occurrences of the selected text, not only the terminal one; intended filenames contain it once at the end. downstream code expects canonical ligandfile/proteinfile columns. prototype/full evaluation csvs are renamed; posebusters already has those columns and is identified by ligandname. unknown schemas fail early rather than constructing wrong paths. this function centralizes the fixed directory convention. protein comes first in its return tuple because posebusters conditions validity checks on the pocket; rmsd itself only needs the two ligand paths.",
      "title": "evaluation/evaluation.py",
      "url": "walkthrough-evaluation.html#dataset-schemas-and-paths"
    },
    {
      "excerpt": "PoseBusters checks whether a docked pose is physically/chemically plausible—such as internal geometry and clashes—using predicted ligand, true ligand, and conditioning protein. The dock configuration selects its docking…",
      "heading": "PoseBusters integration",
      "text": "evaluation/evaluation.py posebusters integration posebusters checks whether a docked pose is physically/chemically plausible—such as internal geometry and clashes—using predicted ligand, true ligand, and conditioning protein. the dock configuration selects its docking test suite. a missing package is treated as a configuration error and exits; a per-complex runtime failure returns an empty dataframe so evaluation can continue.",
      "title": "evaluation/evaluation.py",
      "url": "walkthrough-evaluation.html#posebusters-integration"
    },
    {
      "excerpt": "One CSV row is the unit of evaluation. Paths and ID are derived once and used for diagnostics and result fields. The protein is required only for PoseBusters, not ligand RMSD. Collecting all missing paths gives one actio…",
      "heading": "Evaluating one complex",
      "text": "evaluation/evaluation.py evaluating one complex one csv row is the unit of evaluation. paths and id are derived once and used for diagnostics and result fields. the protein is required only for posebusters, not ligand rmsd. collecting all missing paths gives one actionable warning instead of failing on the first. an unparseable reference invalidates all comparisons. at least one predicted record must parse inside the requested prefix. the conventional docking success threshold is rmsd below 2 a; below 1 a is stricter. nan never counts as success. rmsd is stored to four decimals while booleans are computed from the unrounded value. note that after invalid sdf records are skipped, ranks are renumbered consecutively from the parseable list rather than preserving original record indices. posebusters is run once on the whole predicted sdf, then rows are paired by position with evaluated poses. pb-valid is true only if every boolean posebusters test in that row passes; individual booleans are also copied unless their name would overwrite an existing result key. empty check output records unknown validity as none, distinct from a real failure (false).",
      "title": "evaluation/evaluation.py",
      "url": "walkthrough-evaluation.html#evaluating-one-complex"
    },
    {
      "excerpt": "The evaluator iterates pandas rows in CSV order and reports progress every 50. allresults is flattened because one complex may yield several pose rows. A complex is skipped only if it yields no pose rows; an RMSD-NaN pos…",
      "heading": "Evaluating a dataset and summarizing",
      "text": "evaluation/evaluation.py evaluating a dataset and summarizing the evaluator iterates pandas rows in csv order and reports progress every 50. allresults is flattened because one complex may yield several pose rows. a complex is skipped only if it yields no pose rows; an rmsd-nan pose still counts as evaluated. returning an empty dataframe is not itself a process failure. otherwise pandas forms the union of keys, so posebusters columns absent from some rows become missing values. top-1 means the first successfully loaded output record. in pandas, boolean mean is the fraction true; median ignores nan rmsds. pb none becomes missing and is normally excluded from its mean, so failed checks need not count as invalid in the denominator. best-of-n is an oracle metric: for each complex it selects the pose with lowest reference rmsd, not the one the model ranked highest. this measures whether a good pose exists in the first n. pb validity is then reported for that rmsd-selected pose. parent directories are created recursively and the flat per-pose table is written without a pandas index column. returning it also makes the function usable programmatically.",
      "title": "evaluation/evaluation.py",
      "url": "walkthrough-evaluation.html#evaluating-a-dataset-and-summarizing"
    },
    {
      "excerpt": "Raw-description formatting preserves the multiline usage example. Dataset choices constrain directory/schema logic. The negative flag makes PoseBusters on by default, and type=Path converts only that path argument automa…",
      "heading": "Command-line interface and entry point",
      "text": "evaluation/evaluation.py command-line interface and entry point raw-description formatting preserves the multiline usage example. dataset choices constrain directory/schema logic. the negative flag makes posebusters on by default, and type=path converts only that path argument automatically. all paths are relative to the current working directory, so the command must run from the repository root or evaluate.sh's synthetic root. missing input data is fatal. missing predictions are only a warning so per-complex skipping logic can report coverage. the default csv sits directly under results/, separate from per-complex pose directories. not args.nopbvalid converts the negative flag into the positive function argument. the module guard keeps import side-effect free except for logging configuration.",
      "title": "evaluation/evaluation.py",
      "url": "walkthrough-evaluation.html#command-line-interface-and-entry-point"
    },
    {
      "excerpt": "GetBestRMS optimally aligns the predicted ligand to the reference. For blind docking, that can hide translation/rotation errors relative to the protein pocket and produce much more optimistic numbers than fixed-frame RMS…",
      "heading": "Gotchas / notes",
      "text": "evaluation/evaluation.py gotchas / notes getbestrms optimally aligns the predicted ligand to the reference. for blind docking, that can hide translation/rotation errors relative to the protein pocket and produce much more optimistic numbers than fixed-frame rmsd. aggregate percentages use only complexes that produced result rows. missing predictions/references are skipped rather than counted as failures, so coverage must be reported alongside success rates. posebusters reads the entire prediction sdf, while rmsd loading stops at topn and drops invalid records. pairing posebusters rows to compacted rmsd rows by position can misalign results when early sdf records are invalid; it also assumes posebusters preserves record order. topn is not validated as positive. zero or negative values load no poses and cause every complex to be skipped. if posebusters fails and sets pbvalid=none, pandas means generally exclude those missing values instead of counting them as invalid. best-of-n uses idxmin; groups whose rmsds are all nan may cause problematic selection depending on pandas behavior. the module docstring names updatedevaluation.py, while the actual repository path is evaluation/evaluation.py; this is a naming artifact, not a second file. why can posebusters validity become paired with the wrong rmsd pose? how can skipped files inflate a reported success percentage? why is pbvalid=none not equivalent to a failed posebusters check?",
      "title": "evaluation/evaluation.py",
      "url": "walkthrough-evaluation.html#gotchas--notes"
    },
    {
      "excerpt": "This command converts KarmaDock's repeat-by-repeat outputs into the evaluator's one-file-per-complex format. It ranks repeated predictions by mixture-density-network (MDN) score and writes a multi-record SDF with the hig…",
      "heading": "Purpose",
      "text": "scripts/convertkarmadocktoseminar.py purpose this command converts karmadock's repeat-by-repeat outputs into the evaluator's one-file-per-complex format. it ranks repeated predictions by mixture-density-network (mdn) score and writes a multi-record sdf with the highest-scoring pose first. why must repeated karmadock predictions be combined into one sdf per complex? why is mdn score used before writing the output records? what would be lost if the converter wrote only the highest-scoring molecule?",
      "title": "scripts/convertkarmadocktoseminar.py",
      "url": "walkthrough-convert-karmadock-to-seminar.html#purpose"
    },
    {
      "excerpt": "runinfer.sh calls it for raw, force-field-corrected, and alignment-corrected outputs after upstream liganddocking.py. It reads /.csv and pose SDFs, a dataset CSV, --mode, and optional --nrepeat (default 3), then writes /…",
      "heading": "How it fits in",
      "text": "scripts/convertkarmadocktoseminar.py how it fits in runinfer.sh calls it for raw, force-field-corrected, and alignment-corrected outputs after upstream liganddocking.py. it reads /.csv and pose sdfs, a dataset csv, --mode, and optional --nrepeat (default 3), then writes /pred.sdf in evaluator rank order. why does runinfer.sh invoke this converter three times? how do --mode and --nrepeat affect the produced sdf? why does the converter also need the dataset csv after docking is complete?",
      "title": "scripts/convertkarmadocktoseminar.py",
      "url": "walkthrough-convert-karmadock-to-seminar.html#how-it-fits-in"
    },
    {
      "excerpt": "pandas reads scores; RDKit parses/writes SDF molecular records and coordinates. choices rejects mistyped modes. The two inputs are validated and output is safely created. Each repeat score table contributes repeat/score…",
      "heading": "Walkthrough",
      "text": "scripts/convertkarmadocktoseminar.py walkthrough pandas reads scores; rdkit parses/writes sdf molecular records and coordinates. choices rejects mistyped modes. the two inputs are validated and output is safely created. each repeat score table contributes repeat/score pairs per complex. an mdn represents interaction-distance likelihoods as a mixture distribution; this scalar score ranks poses with higher treated as better. missing individual repeats are tolerated, but no repeat tables is fatal. descending score determines sdf record order. if a complex has no scores, repeats receive tied zero scores and python's stable sort retains repeat order. ff minimization relaxes geometry under a molecular mechanics force field. alignment correction superimposes coordinates into a reference frame. missing corrected files fall back to raw poses; missing/invalid records are skipped. explicit hydrogens and three provenance properties are retained. closing flushes output. empty files are removed so existence means at least one pose. the summary counts complexes, not pose records. what happens when one repeat csv is missing versus when every repeat csv is missing? how is a complex with no score rows ranked, and why is the result deterministic? why does the converter retain karmadockscore, karmadockrepeat, and karmadockmode properties?",
      "title": "scripts/convertkarmadocktoseminar.py",
      "url": "walkthrough-convert-karmadock-to-seminar.html#walkthrough"
    },
    {
      "excerpt": "Evaluator rank is file order, so descending score and write order are inseparable. A corrected directory can contain raw fallback coordinates; KarmaDockMode records the requested mode, not the fallback. If any score exis…",
      "heading": "Gotchas / notes",
      "text": "scripts/convertkarmadocktoseminar.py gotchas / notes evaluator rank is file order, so descending score and write order are inseparable. a corrected directory can contain raw fallback coordinates; karmadockmode records the requested mode, not the fallback. if any score exists for a complex, repeats absent from its score rows are not considered even if their pose files exist. rdkit parse failures are skipped instead of aborting conversion. why can changing write order change evaluation without changing any coordinates? why does karmadockmode=ffcorrected not prove that the stored coordinates were force-field corrected? how can an existing pose be omitted when its repeat has no score row for that complex?",
      "title": "scripts/convertkarmadocktoseminar.py",
      "url": "walkthrough-convert-karmadock-to-seminar.html#gotchas--notes"
    },
    {
      "excerpt": "This command converts the seminar's flat file layout into the per-complex layout required by upstream KarmaDock preprocessing. It copies rather than chemically transforms structures, so only directory and file naming cha…",
      "heading": "Purpose",
      "text": "scripts/convertseminartokarmadock.py purpose this command converts the seminar's flat file layout into the per-complex layout required by upstream karmadock preprocessing. it copies rather than chemically transforms structures, so only directory and file naming change. why does this converter copy structures instead of modifying molecular chemistry? what invariant does the converter establish for each usable complex? why must ligand and protein files be treated as a pair?",
      "title": "scripts/convertseminartokarmadock.py",
      "url": "walkthrough-convert-seminar-to-karmadock.html#purpose"
    },
    {
      "excerpt": "runtrain.sh, runfulltrain.sh, and runinfer.sh call it before upstream pocket extraction and graph generation. It takes --csv, --srcdir, and --outdir, then writes //ligand.sdf and protein.pdb.",
      "heading": "How it fits in",
      "text": "scripts/convertseminartokarmadock.py how it fits in runtrain.sh, runfulltrain.sh, and runinfer.sh call it before upstream pocket extraction and graph generation. it takes --csv, --srcdir, and --outdir, then writes //ligand.sdf and protein.pdb. why is this converter called before pocket extraction and graph generation? how does the shared complexrecords helper reduce pipeline inconsistency? what are the roles of --srcdir and --outdir?",
      "title": "scripts/convertseminartokarmadock.py",
      "url": "walkthrough-convert-seminar-to-karmadock.html#how-it-fits-in"
    },
    {
      "excerpt": "shutil.copy2 copies contents and preserves metadata where possible; the shared CSV helper guarantees consistent IDs. All paths are mandatory. The two inputs are validated before output work; sys.exit(string) reports the…",
      "heading": "Walkthrough",
      "text": "scripts/convertseminartokarmadock.py walkthrough shutil.copy2 copies contents and preserves metadata where possible; the shared csv helper guarantees consistent ids. all paths are mandatory. the two inputs are validated before output work; sys.exit(string) reports the error and exits nonzero. reruns can reuse the output directory. a complex is usable only when both molecular partners exist; incomplete pairs are counted and skipped. each successful complex gets exactly the names upstream expects. the summary exposes data loss without making a few preprocessing failures fatal. the module guard prevents copying when imported. why are both input paths checked before the output directory is created? what happens when one molecular partner is absent for a csv record? why is the final converted summary operationally important?",
      "title": "scripts/convertseminartokarmadock.py",
      "url": "walkthrough-convert-seminar-to-karmadock.html#walkthrough"
    },
    {
      "excerpt": "Missing pairs do not cause a nonzero exit; downstream graph counts and the summary must be checked. Existing target files are overwritten, while unrelated stale output directories are retained. Pocket extraction and grap…",
      "heading": "Gotchas / notes",
      "text": "scripts/convertseminartokarmadock.py gotchas / notes missing pairs do not cause a nonzero exit; downstream graph counts and the summary must be checked. existing target files are overwritten, while unrelated stale output directories are retained. pocket extraction and graph construction are later upstream steps, not part of this converter. why can a successful process exit still represent incomplete conversion? what stale-state risk comes from using existok=true on reruns? why should users not expect graph files after this command finishes?",
      "title": "scripts/convertseminartokarmadock.py",
      "url": "walkthrough-convert-seminar-to-karmadock.html#gotchas--notes"
    },
    {
      "excerpt": "This is the prototype batch-scoring driver. It evaluates every available P1/P2/P3 × raw/FF/aligned result directory with the same prototest dataset interface, then copies the P2 raw CSV to the repository's headline filen…",
      "heading": "Purpose",
      "text": "scripts/evaluate.sh purpose this is the prototype batch-scoring driver. it evaluates every available p1/p2/p3 × raw/ff/aligned result directory with the same prototest dataset interface, then copies the p2 raw csv to the repository's headline filename. why does this wrapper evaluate a 3×3 matrix instead of calling the evaluator only once? why is the p2 raw csv copied to prototestevaluation.csv? why is pose generation intentionally absent from this script?",
      "title": "scripts/evaluate.sh",
      "url": "walkthrough-evaluate.html#purpose"
    },
    {
      "excerpt": "condor/evaluate.sub invokes it after the three prototype inference jobs. It expects evaluation/evaluation.py, a prototest.csv, reference structures, and pose trees under results/. It writes results/evaluation.csv and opt…",
      "heading": "How it fits in",
      "text": "scripts/evaluate.sh how it fits in condor/evaluate.sub invokes it after the three prototype inference jobs. it expects evaluation/evaluation.py, a prototest.csv, reference structures, and pose trees under results/. it writes results/evaluation.csv and optionally results/prototestevaluation.csv. there are no positional parameters or custom environment variables. what must inference jobs produce before condor/evaluate.sub can use this wrapper? why does the wrapper probe both repository-style and flattened locations? what outputs can be absent without making the entire wrapper fail?",
      "title": "scripts/evaluate.sh",
      "url": "walkthrough-evaluate.html#how-it-fits-in"
    },
    {
      "excerpt": "Strict Bash mode makes missing paths and failed evaluations stop the job. Unlike training/inference wrappers it does not use set -x, so only explicit messages and evaluator logs appear. The script assumes the current dir…",
      "heading": "Walkthrough",
      "text": "scripts/evaluate.sh walkthrough strict bash mode makes missing paths and failed evaluations stop the job. unlike training/inference wrappers it does not use set -x, so only explicit messages and evaluator logs appear. the script assumes the current directory is the repository or flattened job root. htcondor can preserve directory trees or flatten transferred items depending on staging context, so firstexisting probes several legitimate locations. its first argument is a test operator (-f file or -d directory); shift leaves only candidates. the three guards convert an empty lookup result into a clear fatal message. evaluation.py has fixed relative roots (data/ and results/) and derives paths from the dataset name. rather than move data, this function builds a temporary view with symbolic links so any pose directory appears as results/prototest. --nopbvalid deliberately limits this batch comparison to rmsd, avoiding expensive posebusters checks. the parenthesized cd does not alter the caller's directory. the nested loops define a 3×3 matrix. each suffix:label token is split with bash parameter expansion: %%: removes the colon and everything after it, while ##: removes everything through the last colon. missing pose variants are skipped rather than failing the entire matrix. the final copy creates the conventional default result name while retaining the explicitly labeled source csv. why does evalone construct a temporary directory of symbolic links? how does the suffix:label loop token produce both a directory path and an output filename? why is --nopbvalid used for this batch comparison?",
      "title": "scripts/evaluate.sh",
      "url": "walkthrough-evaluate.html#walkthrough"
    },
    {
      "excerpt": "Temporary directories are removed only at the end of a successful evalone; there is no trap, so an interrupted/failed evaluation can leave a directory in the system temp area. PoseBusters is always disabled by this wrapp…",
      "heading": "Gotchas / notes",
      "text": "scripts/evaluate.sh gotchas / notes temporary directories are removed only at the end of a successful evalone; there is no trap, so an interrupted/failed evaluation can leave a directory in the system temp area. posebusters is always disabled by this wrapper even though it is enabled by default in the python evaluator. the wrapper always exposes data to the evaluator as prototest, even when iterating different model pipelines; that is correct because the structures are the same and only predictions change. re-running overwrites existing evaluation csvs and the headline copy. why can a failed evaluation leave temporary files behind? why is exposing every pipeline as dataset prototest correct rather than misleading? what happens to existing csvs when this wrapper is rerun?",
      "title": "scripts/evaluate.sh",
      "url": "walkthrough-evaluate.html#gotchas--notes"
    },
    {
      "excerpt": "This is the single-GPU full-dataset analogue of runtrain.sh. Because the full structures and generated graphs are large, it reads and writes them through a home directory mounted into the Docker job rather than transferr…",
      "heading": "Purpose",
      "text": "scripts/runfulltrain.sh purpose this is the single-gpu full-dataset analogue of runtrain.sh. because the full structures and generated graphs are large, it reads and writes them through a home directory mounted into the docker job rather than transferring them with every htcondor submission. why does full-data training use home-mounted storage instead of transferring all artifacts into each job sandbox? how is this script related to runtrain.sh? what limitation remains despite supporting the full dataset?",
      "title": "scripts/runfulltrain.sh",
      "url": "walkthrough-run-full-train.html#purpose"
    },
    {
      "excerpt": "condor/fulltrainscratch.sub invokes it as scratch; finetune is also implemented even though no listed submit file uses that mode. It reads $FULLDATADIR/{fulltrain,fullval}.csv and matching structure directories, defaulti…",
      "heading": "How it fits in",
      "text": "scripts/runfulltrain.sh how it fits in condor/fulltrainscratch.sub invokes it as scratch; finetune is also implemented even though no listed submit file uses that mode. it reads $fulldatadir/{fulltrain,fullval}.csv and matching structure directories, defaulting to $home/reprotest/fulldata. it persists converted data, graphs, w&b state, and checkpoints below $fullworkdir, defaulting to $home/reprotest/workfull. which mode does the listed condor submit file use, and what other mode is available manually? why are fulltrain and fullval supplied as separate inputs? what does $fullworkdir persist across jobs?",
      "title": "scripts/runfulltrain.sh",
      "url": "walkthrough-run-full-train.html#how-it-fits-in"
    },
    {
      "excerpt": "The strict/debug Bash options have the same roles as in runtrain.sh. ${VAR:-default} makes the paths relocatable. HTCondor stages the scripts directory beneath the sandbox working directory even though the executable its…",
      "heading": "Walkthrough",
      "text": "scripts/runfulltrain.sh walkthrough the strict/debug bash options have the same roles as in runtrain.sh. ${var:-default} makes the paths relocatable. htcondor stages the scripts directory beneath the sandbox working directory even though the executable itself may appear flattened, so the code deliberately uses $pwd/scripts rather than the executable's directory. the function removes duplication between the curated train and validation splits. local keeps its variables from leaking into global script state. both splits get their own converted structures and graph directory. the separate validation graphs are important: full validation is curated rather than randomly carved from training. a bash array preserves each argument as a distinct shell word, even when a path contains spaces. every mode shares the explicit leak-resistant validation split, batch settings, resume behavior, and w&b logging. w&b credentials are expected via wandbapikey or the mounted home .netrc; train.py falls back to offline logging otherwise. this is the same two-stage objective as the prototype driver: mdn-only scoring first, then combined scoring and docking-coordinate learning. the explicit validation csv means all available fulltrain graphs remain training data and all available fullval graphs remain validation data. the alternative initializes from released weights and trains a single combined-loss stage. the mode check is intentionally late because preprocessing happens before it; valid submit files always pass a known mode. why does the preprocess function use a marker independently for each split? how does the common bash array improve correctness when building trainer commands? how do scratch and fine-tune differ in their initialization and number of stages?",
      "title": "scripts/runfulltrain.sh",
      "url": "walkthrough-run-full-train.html#walkthrough"
    },
    {
      "excerpt": "The script assumes HTCondor mounted the user's home (+WantGPUHomeMounted = true in the submit file). Without it, defaults may point at unavailable container paths. Preprocessing markers have the same stale-marker risk as…",
      "heading": "Gotchas / notes",
      "text": "scripts/runfulltrain.sh gotchas / notes the script assumes htcondor mounted the user's home (+wantgpuhomemounted = true in the submit file). without it, defaults may point at unavailable container paths. preprocessing markers have the same stale-marker risk as runtrain.sh. mode validation occurs after both splits are preprocessed, so an invalid mode can still perform expensive preprocessing before exiting. the comments explicitly call one-gpu full training impractical; this driver is functional but does not exploit multiple gpus. what fails if the htcondor job does not mount the user's home? why can an invalid mode still consume substantial time before producing its error? why can a .preprocessed marker become misleading?",
      "title": "scripts/runfulltrain.sh",
      "url": "walkthrough-run-full-train.html#gotchas--notes"
    },
    {
      "excerpt": "This script runs the complete pose-generation path for one dataset/checkpoint pair. It converts structures, creates pockets and graphs, calls upstream KarmaDock docking, and exports raw, force-field-minimized, and alignm…",
      "heading": "Purpose",
      "text": "scripts/runinfer.sh purpose this script runs the complete pose-generation path for one dataset/checkpoint pair. it converts structures, creates pockets and graphs, calls upstream karmadock docking, and exports raw, force-field-minimized, and alignment-corrected poses in seminar format. why is this script described as a complete pose-generation path rather than an evaluation path? why are raw, force-field, and aligned outputs all exported? what makes one invocation specific to a model experiment?",
      "title": "scripts/runinfer.sh",
      "url": "walkthrough-run-infer.html#purpose"
    },
    {
      "excerpt": "Five listed inference submit files call it with three positional arguments: dataset name, checkpoint path/basename, and output tag. It reads .csv and / from the current job sandbox, creates temporary intermediate data, a…",
      "heading": "How it fits in",
      "text": "scripts/runinfer.sh how it fits in five listed inference submit files call it with three positional arguments: dataset name, checkpoint path/basename, and output tag. it reads .csv and / from the current job sandbox, creates temporary intermediate data, and writes results//{,ff,align}/pred.sdf. scoring happens later in evaluate.sh or evaluation.py. why do inference submit files pass a checkpoint basename in some cases and an absolute path in others? what persistent output contract does the script provide to later scoring? why are intermediate graphs not part of that output contract?",
      "title": "scripts/runinfer.sh",
      "url": "walkthrough-run-infer.html#how-it-fits-in"
    },
    {
      "excerpt": "Strict mode stops partial pipelines, and tracing makes the cluster log reproducible. Unlike runtrain.sh, these positional variables do not use ${1:?...}; with set -u, a missing one still fails immediately, but without th…",
      "heading": "Walkthrough",
      "text": "scripts/runinfer.sh walkthrough strict mode stops partial pipelines, and tracing makes the cluster log reproducible. unlike runtrain.sh, these positional variables do not use ${1:?...}; with set -u, a missing one still fails immediately, but without the custom usage text. the case distinguishes absolute paths (/) from transferred checkpoint basenames. relative checkpoints are resolved before the script later changes directory. mktemp -d isolates a run; the quoted trap removes it on normal exit or most errors. only exported results persist. the first command performs naming/layout conversion only. upstream preprocessing then identifies the binding pocket and graph generation turns atoms/residues and their relationships into dgl graph tensors. the count is a quick coverage diagnostic, not a validation that every expected id succeeded. this is upstream karmadock's inference entry point. --docking true predicts coordinates, --scoring true evaluates poses with the mdn head, and --correct true requests all three pose forms. the raw network output is “uncorrected”; ff correction performs molecular mechanics relaxation; alignment correction superimposes coordinates into a reference frame. the fixed random seed makes stochastic start poses/repeats reproducible within the same software/hardware environment. the function packages the repeated converter invocation. ${out#$submit/} removes a leading submit-directory prefix only for cleaner logging. the converter ranks docking repeats by score, so each resulting multi-record sdf is best-first. three calls create parallel result trees for fair postprocessing comparisons. why is a relative checkpoint resolved before changing into karmadock's utility directory? what distinct work do preprocessing.py, generategraph.py, and liganddocking.py perform? how does exportvariant turn native repeat outputs into fair comparable result trees?",
      "title": "scripts/runinfer.sh",
      "url": "walkthrough-run-infer.html#walkthrough"
    },
    {
      "excerpt": "Temporary graphs and native KarmaDock output are always deleted at exit, so debugging them requires changing the script or copying them before exit. The graph-count and pose-count commands use pipelines under pipefail; t…",
      "heading": "Gotchas / notes",
      "text": "scripts/runinfer.sh gotchas / notes temporary graphs and native karmadock output are always deleted at exit, so debugging them requires changing the script or copying them before exit. the graph-count and pose-count commands use pipelines under pipefail; the explicit 2>/dev/null hides “no glob matches” diagnostics, but an empty pose directory can still make that command substitution fail under strict pipeline semantics. the current seminarcsv.py supports ligandfilename or the four full metadata columns, but the checked-in data/posebustersfiltered.csv instead has ligandfile. therefore the listed posebusters submit route is inconsistent with this converter as written and can raise a missing-column error before inference. alignment uses the reference pose/frame and should be interpreted carefully as a postprocessing diagnostic, not a blind-docking output independent of the answer. why are temporary native docking outputs difficult to inspect after a failure? how can a diagnostic pose-count command itself stop the script? why should aligned success not be interpreted as reference-independent blind docking accuracy?",
      "title": "scripts/runinfer.sh",
      "url": "walkthrough-run-infer.html#gotchas--notes"
    },
    {
      "excerpt": "This is the self-contained prototype training driver. It converts and preprocesses prototrain, then either runs the two-stage from-scratch protocol (P2) or fine-tunes the released checkpoint (P3), using the single-proces…",
      "heading": "Purpose",
      "text": "scripts/runtrain.sh purpose this is the self-contained prototype training driver. it converts and preprocesses prototrain, then either runs the two-stage from-scratch protocol (p2) or fine-tunes the released checkpoint (p3), using the single-process train.py loop. why does the prototype training driver include preprocessing as well as trainer invocation? how do the two advertised modes correspond to the benchmark pipelines? why is this wrapper described as single-process?",
      "title": "scripts/runtrain.sh",
      "url": "walkthrough-run-train.html#purpose"
    },
    {
      "excerpt": "condor/p2trainscratch.sub calls it with scratch; condor/p3finetune.sub calls it with finetune. It expects prototrain.csv, prototrain/, and scripts/ in the current HTCondor sandbox, plus upstream KarmaDock at /app/KarmaDo…",
      "heading": "How it fits in",
      "text": "scripts/runtrain.sh how it fits in condor/p2trainscratch.sub calls it with scratch; condor/p3finetune.sub calls it with finetune. it expects prototrain.csv, prototrain/, and scripts/ in the current htcondor sandbox, plus upstream karmadock at /app/karmadock. it writes reusable preprocessing under worktrain/ and checkpoints/logs under ckpt/. which condor files select each training mode? what must htcondor stage into the current sandbox before this wrapper starts? which outputs are intended to survive the run?",
      "title": "scripts/runtrain.sh",
      "url": "walkthrough-run-train.html#how-it-fits-in"
    },
    {
      "excerpt": "The env shebang locates Bash through the environment. set -e aborts on an unhandled failing command, -u rejects unset variables, and pipefail makes a pipeline fail if any component fails. set -x echoes expanded commands…",
      "heading": "Walkthrough",
      "text": "scripts/runtrain.sh walkthrough the env shebang locates bash through the environment. set -e aborts on an unhandled failing command, -u rejects unset variables, and pipefail makes a pipeline fail if any component fails. set -x echoes expanded commands for cluster debugging. ${1:?...} makes the first positional argument mandatory. ${pythonpath:-} is safe under -u and exposes upstream karmadock imports to local python programs. the work directory is persistent because an evicted/rescheduled cluster job can reuse graphs and --resume checkpoints. the converter creates karmadock's input layout; upstream preprocessing.py extracts/normalizes pockets, and generategraph.py serializes graph objects as .dgl files. parentheses make the cd local to a subshell. python's -u gives unbuffered logs. the sentinel is touched only if all preprocessing commands succeeded under set -e. stage 1 starts from random weights (--initmodel \"\") and sets posr=0, so the coordinate/rmsd term is multiplied by zero and training focuses on the mdn scoring objective. gradient accumulation performs one optimizer update after 16 physical batches of 4, approximating an effective batch of 64 without storing all 64 examples simultaneously. early stopping may end the nominal 1000 epochs after 70 non-improving validations. stage.done prevents a completed stage 1 from being rerun. stage 2 initializes from the best stage-1 checkpoint and enables coordinate learning with posr=1. it uses a smaller learning rate, stronger weight decay, shorter patience, and 0.05 a gaussian coordinate jitter as augmentation. karmadock's pose refinement “recycles” its equivariant coordinate-update network three times inside the upstream model; this script does not implement that architecture, but stage 2 provides the rmsd signal that trains it. fine-tuning skips stage 1 and begins with the released screening weights, optimizing both losses. the final branch rejects any value other than the two advertised modes with exit status 2. why is worktrain persistent and protected by a .preprocessed marker? how does stage 1 obtain effective batch 64 without loading 64 complexes simultaneously? why does stage 2 initialize from stage 1 and change posr from 0 to 1? how does fine-tuning differ from simply resuming stage 2?",
      "title": "scripts/runtrain.sh",
      "url": "walkthrough-run-train.html#walkthrough"
    },
    {
      "excerpt": "The .preprocessed and stage.done markers assert completion without checking whether files were later deleted or corrupted; remove the marker to force regeneration. --resume only resumes if last.pt exists. A completed Sta…",
      "heading": "Gotchas / notes",
      "text": "scripts/runtrain.sh gotchas / notes the .preprocessed and stage.done markers assert completion without checking whether files were later deleted or corrupted; remove the marker to force regeneration. --resume only resumes if last.pt exists. a completed stage 2 has no separate marker and will be invoked again, although its checkpoint may resume near the previous end. the “effective batch 64” statement is exact for full accumulation windows; a final partial window is still divided by 16 in train.py, so its gradient is smaller. why can marker files falsely indicate a complete training prerequisite? why can a completed stage 2 be invoked again even though stage 1 is skipped? why is a final partial accumulation window underweighted?",
      "title": "scripts/runtrain.sh",
      "url": "walkthrough-run-train.html#gotchas--notes"
    },
    {
      "excerpt": "This module gives the pipeline one definition of a complex identifier: the stable name for one protein-pocket/ligand pair. It hides the difference between the prototype CSV schema, which already contains filenames, and t…",
      "heading": "Purpose",
      "text": "scripts/seminarcsv.py purpose this module gives the pipeline one definition of a complex identifier: the stable name for one protein-pocket/ligand pair. it hides the difference between the prototype csv schema, which already contains filenames, and the full-data schema, which contains four metadata fields from which filenames must be constructed. why must every pipeline component share one definition of complex id? what schema difference does this module hide from its callers? why does the output preserve first-seen order?",
      "title": "scripts/seminarcsv.py",
      "url": "walkthrough-seminar-csv.html#purpose"
    },
    {
      "excerpt": "Both converters import complexrecords; both trainers import complexids. The input is a CSV path. The outputs are ordered (complexid, ligandfilename, proteinfilename) tuples or just IDs, which must exactly match the stems…",
      "heading": "How it fits in",
      "text": "scripts/seminarcsv.py how it fits in both converters import complexrecords; both trainers import complexids. the input is a csv path. the outputs are ordered (complexid, ligandfilename, proteinfilename) tuples or just ids, which must exactly match the stems of karmadock's .dgl graph files. why do both converters import complexrecords? why do trainers use complexids instead of independently parsing csv fields? what relationship must hold between a returned id and preprocessing output?",
      "title": "scripts/seminarcsv.py",
      "url": "walkthrough-seminar-csv.html#how-it-fits-in"
    },
    {
      "excerpt": "The constant fixes both the fields and their order. Changing that order would change paths and graph names. This private helper derives the ID from either refined seminar or shorter ligand names. replace is not restricte…",
      "heading": "Walkthrough",
      "text": "scripts/seminarcsv.py walkthrough the constant fixes both the fields and their order. changing that order would change paths and graph names. this private helper derives the id from either refined seminar or shorter ligand names. replace is not restricted to the end, although valid input uses these as suffixes. this generator chooses the prototype schema by presence of ligandfilename; otherwise it transposes the four metadata columns into row tuples, joins them with underscores, and constructs refined filenames. centralizing the branch prevents conversion and training from deriving different ids. dtype=str prevents a residue such as 210 becoming 210.0. the set gives fast duplicate checks while the list preserves first-seen order; a repeated complex is intentionally returned once. this projects each triple down to the graph stem. underscore-prefixed variables are deliberately unused. how does rowstotriples decide which schema branch to use? why is pandas called with dtype=str? how do the seen set and records list serve different purposes?",
      "title": "scripts/seminarcsv.py",
      "url": "walkthrough-seminar-csv.html#walkthrough"
    },
    {
      "excerpt": "Schema selection checks only ligandfilename; a partial schema fails later with a missing-column error. The checked-in PoseBusters CSV uses ligandfile, not either schema recognized here, so it is not currently compatible…",
      "heading": "Gotchas / notes",
      "text": "scripts/seminarcsv.py gotchas / notes schema selection checks only ligandfilename; a partial schema fails later with a missing-column error. the checked-in posebusters csv uses ligandfile, not either schema recognized here, so it is not currently compatible with this helper. stripsuffix uses replacement rather than strict terminal suffix removal. why does a partial prototype schema fail less clearly than a fully unknown schema check might? why is the checked-in posebusters csv incompatible with the helper? what is risky about using replace in stripsuffix?",
      "title": "scripts/seminarcsv.py",
      "url": "walkthrough-seminar-csv.html#gotchas--notes"
    },
    {
      "excerpt": "This file supplies the single-process training loop missing from upstream KarmaDock. It connects precomputed molecular graphs to KarmaDock's existing forward pass, combines its pose-coordinate and MDN scoring losses, per…",
      "heading": "Purpose",
      "text": "scripts/train.py purpose this file supplies the single-process training loop missing from upstream karmadock. it connects precomputed molecular graphs to karmadock's existing forward pass, combines its pose-coordinate and mdn scoring losses, performs optimization/validation, and writes resumable plus best-validation checkpoints. what capability does this file add that upstream karmadock does not provide? why does the trainer combine rmsd-related and mdn losses rather than implementing a new docking architecture? why are both a best checkpoint and a resume checkpoint written?",
      "title": "scripts/train.py",
      "url": "walkthrough-train.html#purpose"
    },
    {
      "excerpt": "runtrain.sh uses it for prototype from-scratch and fine-tuning runs; runfulltrain.sh uses it for the large curated train/validation split. It imports model/dataset utilities from /app/KarmaDock and CSV IDs from scripts/s…",
      "heading": "How it fits in",
      "text": "scripts/train.py how it fits in runtrain.sh uses it for prototype from-scratch and fine-tuning runs; runfulltrain.sh uses it for the large curated train/validation split. it imports model/dataset utilities from /app/karmadock and csv ids from scripts/seminarcsv.py. required cli inputs are --csv, --graphdir, and --outdir; optional inputs include a separate validation csv/graph directory, an initialization checkpoint, resume state, optimization hyperparameters, coordinate jitter, and w&b settings. outputs are karmadockteam002.pkl (best), last.pt (resume), trainlog.csv, and optional w&b files in outdir. how do the prototype and full-data wrappers use the same trainer for different data policies? why must --csv, --graphdir, and --outdir come from the orchestration layer? which outputs allow the run to be inspected without w&b?",
      "title": "scripts/train.py",
      "url": "walkthrough-train.html#how-it-fits-in"
    },
    {
      "excerpt": "The file first documents the crucial upstream interface and imports its dependencies: RMSD (root-mean-square deviation) measures average coordinate error in ångströms after matching corresponding ligand atoms; here the u…",
      "heading": "Walkthrough",
      "text": "scripts/train.py walkthrough the file first documents the crucial upstream interface and imports its dependencies: rmsd (root-mean-square deviation) measures average coordinate error in angstroms after matching corresponding ligand atoms; here the upstream training forward pass supplies an rmsd-related pose loss. an mdn (mixture-density network) predicts a probability distribution as a weighted mixture rather than one point value; karmadock uses it to model protein–ligand interaction distances and derive a scoring loss. posr controls whether coordinate learning contributes. the architectural details—gvp/transformer encoding and three recycled equivariant coordinate-refinement passes—remain inside karmadock; “recycling” means feeding an updated pose through the refinement module repeatedly. why does graph availability filtering happen before splitting or loader construction? how do on-the-fly loading and coordinate jitter provide different forms of augmentation? why does the script use dataparallel with only gpu 0? what sequence occurs at a complete gradient-accumulation boundary? why are last.pt and the best-validation checkpoint updated by different rules?",
      "title": "scripts/train.py",
      "url": "walkthrough-train.html#walkthrough"
    },
    {
      "excerpt": "These arguments describe data provenance and lifecycle. An explicit --valcsv is important for leak-proof or curated splits: randomly carving validation examples from training could place closely related protein/ligand sy…",
      "heading": "Argument parsing",
      "text": "scripts/train.py argument parsing these arguments describe data provenance and lifecycle. an explicit --valcsv is important for leak-proof or curated splits: randomly carving validation examples from training could place closely related protein/ligand systems on both sides. storetrue means --resume defaults false and becomes true merely by appearing. the physical batch is what simultaneously occupies memory. gradient accumulation adds gradients from several microbatches before calling optimizer.step; dividing each loss by accumsteps makes a complete window approximate the mean gradient of the larger effective batch. weight decay discourages large parameters. gradient clipping caps the global gradient norm to limit unstable updates. jitter adds small gaussian perturbations to ligand coordinates so the model sees nearby start poses rather than memorizing exact ones. w&b is opt-in and no api key is accepted as a cli flag, reducing the chance that credentials appear in process listings or logs.",
      "title": "scripts/train.py",
      "url": "walkthrough-train.html#argument-parsing"
    },
    {
      "excerpt": "Some molecules fail RDKit parsing or pocket/graph construction, so the loop filters to actual .dgl files instead of failing during training. RandomState(seed) produces a repeatable permutation without depending on NumPy'…",
      "heading": "Selecting usable graphs and splitting data",
      "text": "scripts/train.py selecting usable graphs and splitting data some molecules fail rdkit parsing or pocket/graph construction, so the loop filters to actual .dgl files instead of failing during training. randomstate(seed) produces a repeatable permutation without depending on numpy's global generator. validation gets the first rounded fraction and at least one item. a set makes membership checks cheap; both returned lists retain the filtered dataset's original order within each partition. with an explicit validation csv, no split is performed: the helper only applies the same availability filter to each curated side.",
      "title": "scripts/train.py",
      "url": "walkthrough-train.html#selecting-usable-graphs-and-splitting-data"
    },
    {
      "excerpt": "The upstream dataset loads protein–ligand graph objects. onthefly=True reloads each saved graph and allows KarmaDock's dataset logic to randomize the ligand's starting pose on repeated access—augmentation appropriate for…",
      "heading": "Constructing loaders",
      "text": "scripts/train.py constructing loaders the upstream dataset loads protein–ligand graph objects. onthefly=true reloads each saved graph and allows karmadock's dataset logic to randomize the ligand's starting pose on repeated access—augmentation appropriate for learning to move an initially misplaced ligand into a pocket. the custom loader can pass over invalid/none samples. worker processes prepare batches in parallel, and pinned host memory can speed host-to-cuda transfers. training shuffles; validation does not.",
      "title": "scripts/train.py",
      "url": "walkthrough-train.html#constructing-loaders"
    },
    {
      "excerpt": "CUDA is preferred but CPU remains a functional fallback. The upstream seed helper is expected to seed relevant random generators. W&B is imported lazily so runs without it do not require the package. Missing credentials…",
      "heading": "Startup and optional W&B",
      "text": "scripts/train.py startup and optional w&b cuda is preferred but cpu remains a functional fallback. the upstream seed helper is expected to seed relevant random generators. w&b is imported lazily so runs without it do not require the package. missing credentials switch to offline logs rather than abort training. stage labeling is descriptive: any nonzero posr is labeled docking. a persisted run id lets a genuine checkpoint resume append to the same w&b run. fresh ids combine base, unix timestamp, and pid; parsing the timestamp produces a readable name. vars(args) captures the full experiment configuration. the broad exception handler treats observability as optional: logging failure must not discard an expensive training job. the bare open(...).read() is faithful to the source; unlike the later with, it relies on normal file-object cleanup.",
      "title": "scripts/train.py",
      "url": "walkthrough-train.html#startup-and-optional-wb"
    },
    {
      "excerpt": "An empty validation graph path falls back to the training graph directory. Explicit validation uses all usable training IDs and all usable validation IDs; otherwise one deterministic random split is made. DataParallel wr…",
      "heading": "Data and model initialization",
      "text": "scripts/train.py data and model initialization an empty validation graph path falls back to the training graph directory. explicit validation uses all usable training ids and all usable validation ids; otherwise one deterministic random split is made. dataparallel wraps the model and prefixes checkpoint parameter names with module., matching what upstream inference expects. in this script only gpu 0 is listed, so it standardizes the wrapper/checkpoint format rather than providing multi-gpu scaling. maplocation loads tensors onto the selected device. strict=false tolerates missing/unexpected checkpoint keys, useful across related checkpoints but capable of hiding incompatibilities. adam keeps adaptive first/second-moment estimates for each parameter. two checkpoints serve different purposes: upstream earlystopper writes the best model for inference, while last.pt also contains optimizer state and epoch for exact continuation. resume is strict because it should restore the same architecture, and begins after the saved epoch. the code restores the best score but does not explicitly restore the stopper's count of consecutive bad epochs. a fresh run truncates/creates a simple analysis-friendly csv. a resumed run appends later, preserving earlier rows.",
      "title": "scripts/train.py",
      "url": "walkthrough-train.html#data-and-model-initialization"
    },
    {
      "excerpt": "model.train() enables training behavior such as dropout. Graph batches move to the device. Jitter samples an independent normal displacement with the same tensor shape/device/dtype as ligand positions. Calling .backward(…",
      "heading": "Training and gradient accumulation",
      "text": "scripts/train.py training and gradient accumulation model.train() enables training behavior such as dropout. graph batches move to the device. jitter samples an independent normal displacement with the same tensor shape/device/dtype as ligand positions. calling .backward() accumulates gradients in parameter .grad buffers; pytorch does not clear them automatically. there is no amp/autocast here: all operations use the tensors/model's normal precision, so memory savings from automatic mixed precision are not implemented. at an accumulation boundary, gradients are clipped, adam updates parameters, and buffers are cleared. float(cudatensor) extracts a scalar and synchronizes device/host execution, trading some speed for straightforward logging. a final incomplete window is flushed so samples are not discarded.",
      "title": "scripts/train.py",
      "url": "walkthrough-train.html#training-and-gradient-accumulation"
    },
    {
      "excerpt": "The max prevents division by zero, although zero usable training batches would then report zero training means and is not otherwise fatal here. Validation delegates to upstream code, which is expected to disable gradient…",
      "heading": "Validation and logging",
      "text": "scripts/train.py validation and logging the max prevents division by zero, although zero usable training batches would then report zero training means and is not otherwise fatal here. validation delegates to upstream code, which is expected to disable gradient work/evaluate consistently and return arrays/tensors of losses. means summarize batches, not explicitly individual complexes; weighting therefore follows the upstream return structure. human logs, local csv, and optional remote tracking receive the same metrics. perfcounter is a monotonic high-resolution timer appropriate for elapsed duration.",
      "title": "scripts/train.py",
      "url": "walkthrough-train.html#validation-and-logging"
    },
    {
      "excerpt": "“Lower” validation loss is better. The upstream stopper saves the best model to bestckpt and marks early stop after its patience rule is met. Independently saving last.pt every epoch makes cluster rescheduling recoverabl…",
      "heading": "Checkpointing, early stopping, and shutdown",
      "text": "scripts/train.py checkpointing, early stopping, and shutdown “lower” validation loss is better. the upstream stopper saves the best model to bestckpt and marks early stop after its patience rule is met. independently saving last.pt every epoch makes cluster rescheduling recoverable even when the latest epoch is not the best. successful logged runs record the best loss and attempt to upload the best checkpoint as a model artifact. artifact failure is nonfatal. the module guard prevents training on import.",
      "title": "scripts/train.py",
      "url": "walkthrough-train.html#checkpointing-early-stopping-and-shutdown"
    },
    {
      "excerpt": "The final partial accumulation window is divided by the full accumsteps, so it contributes a smaller-than-full-window gradient rather than being renormalized by its actual microbatch count. Resume restores bestscore but…",
      "heading": "Gotchas / notes",
      "text": "scripts/train.py gotchas / notes the final partial accumulation window is divided by the full accumsteps, so it contributes a smaller-than-full-window gradient rather than being renormalized by its actual microbatch count. resume restores bestscore but not an explicit early-stopper “bad epoch” counter. whether patience fully survives rescheduling depends on the upstream earlystopper implementation, which is not in this repository. strict=false initialization can conceal checkpoint/model key mismatches; the script does not print missing or unexpected keys. empty or nearly empty graph sets are not explicitly rejected. splitavailable also forces one validation id, so a one-graph dataset leaves no training ids. no learning-rate scheduler or amp/autocast is used. these are omissions in the actual implementation, not implicit upstream behavior. why can resuming change the effective early-stopping schedule? what risk accompanies strict=false when loading an initialization checkpoint? why can a very small usable graph set produce invalid training behavior without a clear early error?",
      "title": "scripts/train.py",
      "url": "walkthrough-train.html#gotchas--notes"
    }
  ],
  "lang": "en"
};
