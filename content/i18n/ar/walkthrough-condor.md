# شرح `condor/*.sub` {#condorsub}

## الغرض {#purpose}

تصف هذه الملفات الـ12 مهام المستودع لنظام HTCondor: تدريب النموذج الأولي والبيانات الكاملة، وتجارب المرحلة 2 على GPU واحدة أو عدة وحدات، والاستدلال، والتقييم. لا تنفذ منطق التعلم الآلي بنفسها؛ بل تختار بيئة Docker، وتنقل الملفات، وتحجز الموارد، وتستدعي غلاف shell بوسائط أو متغيرات بيئة، وتوجه المخرجات والسجلات.

## موقعه في المنظومة {#how-it-fits-in}

يرسل المستخدم ملفًا بواسطة `condor_submit <file>.sub`. يطابق HTCondor المهمة مع عامل، وينقل المدخلات المعلنة إلى صندوق مؤقت، ويشغّل برنامج shell المحدد داخل `ahlamloum/karmadock-seminar:v6`، ثم يعيد المخرجات المعلنة. تستدعي مهام الاستدلال `scripts/run_infer.sh`، وتستدعي مهام التدريب أحد أغلفة التدريب، وتستدعي مهام DDP مشغّل المرحلة 2، وتستدعي مهمة التقييم على CPU ‏`scripts/evaluate.sh` بعد وجود التنبؤات.

## شرح الملفات {#walkthrough}

### مفردات ملفات إرسال HTCondor المشتركة {#shared-htcondor-submit-file-vocabulary}

ملف الإرسال قائمة من توجيهات `key = value` يتبعها `queue`. تعني التوجيهات المشتركة هنا ما يأتي:

- يشغّل `universe = docker` البرنامج داخل `docker_image`. توفر الصورة KarmaDock وPython وPyTorch وRDKit وبقية الاعتماديات.
- `executable` هو البرنامج المنقول، ويمرر `arguments` سطر أوامره الموضعي، بينما يحقن `environment` متغيرات البيئة الخاصة بالمهمة.
- تعلن `request_gpus` و`request_cpus` و`request_memory` الموارد المطلوبة للمطابقة والجدولة وتخبر العامل بمقدار التخصيص. هذه القيم لكل مهمة لا لكل عملية.
- `requirements` تعبير ClassAd في HTCondor. تتطلب الملفات نطاق UID الخاص بعلوم الحاسوب في سارلاند، وتستبعد `idun.hpc.uni-saarland.de` الموثقة كعقدة إشكالية. أما `+WantGPUHomeMounted = true` فهي سمة محلية تطلب تثبيت مجلد منزل المرسل داخل الحاوية.
- تقيد `gpus_minimum_capability` و`gpus_minimum_memory` مطابقة GPU أكثر. لا يطلب قدرة حسابية 8.0 أو أحدث وذاكرة 32,000 MB على الأقل إلا تشغيل 2-GPU النهائي.
- يفعّل `should_transfer_files = YES` نقل الملفات. تسرد `transfer_input_files` الشيفرة والبيانات المنسوخة إلى الصندوق، وتنقل المجلدات تكراريًا. ينتظر `when_to_transfer_output = ON_EXIT` انتهاء المهمة طبيعيًا، وتقيد `transfer_output_files` ما يعاد. لا تحتاج المهام التي تحفظ النتائج مباشرة في منزل مثبت إلى نقل مخرجات كبيرة.
- يلتقط `output` الخرج القياسي، و`error` الخطأ القياسي، و`log` سجل دورة حياة HTCondor. يمنع `$(ClusterId)` و`$(ProcId)` المهام من مشاركة أسماء السجلات.
- تنشئ `queue 1` عملية Condor واحدة من الوصف. لا يرتبط ذلك بعدد عمليات DDP؛ إذ يمكن لمهمة Condor واحدة بدء عدة عمليات GPU محلية بواسطة `torchrun`.

### `condor/evaluate.sub` {#evaluate-sub}

```condor
# evaluate.sub  —  the scoring step (run AFTER the p1/p2/p3 docking jobs).
# Runs scripts/evaluate.sh, which calls `evaluation/evaluation.py --dataset proto_test`
# over every pipeline x pose-variant in results/ and writes the *_evaluation.csv files
#
#   condor_submit condor/evaluate.sub
universe      = docker
docker_image  = ahlamloum/karmadock-seminar:v6
requirements  = UidDomain == "cs.uni-saarland.de" && (Machine =!= "idun.hpc.uni-saarland.de")

executable    = scripts/evaluate.sh

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = scripts, evaluation, data/proto_test.csv, data/prototype_model_data/proto_test, results
transfer_output_files   = results

output = logs/evaluate.$(ClusterId).$(ProcId).out
error  = logs/evaluate.$(ClusterId).$(ProcId).err
log    = logs/evaluate.$(ClusterId).log

request_cpus   = 4
request_memory = 16GB

queue 1
```

هذه مهمة CPU الوحيدة، فلا يوجد `request_gpus`. تنقل جميع نتائج النموذج الأولي، وتشغّل مصفوفة RMSD ذات 3×3، وتعيد شجرة `results` الموسعة. تخدم أربع وحدات CPU و16 GB ‏RDKit والتقييم لا الاستدلال. يجب تشغيلها بعد الاستدلال لأن مدخلاتها كلها هي الوضعيات التي تنتجها تلك المهام.

### `condor/full_stage2_2gpu.sub` {#full-stage2-2gpu-sub}

```condor
# full_stage2_2gpu.sub — COPY of full_stage2_mgpu.sub for the 2x A100 DDP Stage-2 run.
# Changes vs the original (which is left untouched):
#   request_gpus 4 -> 2 ; cpus 16 -> 8 ; memory 64GB -> 16GB
#   + GPU selector (cap >=8.0 & >=32 GB) so 2 A100s/L40S/H200 are picked, never a 16 GB OOM card
#   environment S2_BATCH=8 S2_ACCUM=4 -> effective batch 2*8*4 = 64 (paper)
#   executable + transfer use run_full_stage2_ddp_v2.sh (wd 0, patience 70)
# ... prerequisite comments unchanged ...
universe      = docker
docker_image  = ahlamloum/karmadock-seminar:v6
+WantGPUHomeMounted = true
requirements  = UidDomain == "cs.uni-saarland.de" && (Machine =!= "idun.hpc.uni-saarland.de")

executable    = run_full_stage2_ddp_v2.sh
environment   = "S2_BATCH=8 S2_ACCUM=4 STAGE1_CKPT=/home/bdldt_team002/repro_test/work_full/ckpt/p_stage1/karmadock_team002.pkl"

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = train_ddp.py, seminar_csv.py, run_full_stage2_ddp_v2.sh

output = logs/s2_2gpu.$(ClusterId).$(ProcId).out
error  = logs/s2_2gpu.$(ClusterId).$(ProcId).err
log    = logs/s2_2gpu.$(ClusterId).log

request_gpus   = 2
request_cpus   = 8
request_memory = 16GB
gpus_minimum_capability = 8.0
gpus_minimum_memory     = 32000

queue 1
```

هذه تجربة وحدتي GPU النهائية الفعلية. تحافظ تجاوزات البيئة على دفعة عالمية فعلية 64 وتشير مباشرة إلى أفضل نقطة تحقق نهائية للمرحلة 1. يغيّر مشغّل v2 اضمحلال الأوزان وpatience للمرحلة 2. تتجنب قيود GPU الأجهزة ذات 16 GB المعروفة بنفاد الذاكرة. لا تنقل إلا الشيفرة لأن البيانات والرسوم ونقطة التحقق والمخرجات موجودة في المنزل المثبت. يجعل المسار الثابت `/home/bdldt_team002/...` الملف خاصًا بحساب رغم قابلية تجاوز المسارات في المشغّل.

### `condor/full_stage2_mgpu.sub` {#full-stage2-mgpu-sub}

```condor
# full_stage2_mgpu.sub — ISOLATED multi-GPU (DDP) Stage-2 TEST on the full dataset.
# Separate from the prototype submission and from the running full job (167960):
#   - reads ~/repro_test/work_full graphs (read-only) + a Stage-1 snapshot,
#   - writes only to ~/stage2_mgpu_test/work (its own dir).
# ... setup and smoke-test comments unchanged ...
universe      = docker
docker_image  = ahlamloum/karmadock-seminar:v6
+WantGPUHomeMounted = true
requirements  = UidDomain == "cs.uni-saarland.de" && (Machine =!= "idun.hpc.uni-saarland.de")

executable    = run_full_stage2_ddp.sh
# Per-GPU batch 8 x accum 2 x 4 GPUs = effective batch 64. Override here if relocating data:
#   environment = "S2_BATCH=8 S2_ACCUM=2 STAGE1_CKPT=/path/to/stage1.pkl"

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = train_ddp.py, seminar_csv.py, run_full_stage2_ddp.sh
# data, graphs, checkpoints all live on the mounted home -- nothing large is transferred

output = logs/s2mgpu.$(ClusterId).$(ProcId).out
error  = logs/s2mgpu.$(ClusterId).$(ProcId).err
log    = logs/s2mgpu.$(ClusterId).log

request_gpus   = 4
request_cpus   = 16
request_memory = 64GB

queue 1
```

هذه تجربة المرحلة 2 الأصلية على أربع وحدات GPU باستخدام قيم `run_full_stage2_ddp.sh` الافتراضية: دفعة 8 × تجميع 2 × أربع رتب = 64. تطلب CPU وذاكرة أكثر، لكنها لا تفرض قيدًا على جيل GPU أو ذاكرتها بخلاف نسخة 2-GPU. تتوقع الإرسال من مجلد نشر يحوي الملفات الثلاثة المنقولة بأسمائها المجردة ونسخة مرحلة 1 منشأة مسبقًا في المنزل المثبت.

### `condor/full_test_infer.sub` {#full-test-infer-sub}

```condor
# full_test_infer.sub  —  dock the 6,183-complex full_test set with the FULL-DATA
# (multi-GPU Stage-2 trained) checkpoint. Produces the 3 pose variants
# (uncorrected / FF / align) in results/full_scratch/full_test{,_ff,_align}/.
# ... prerequisite comments unchanged ...
universe      = docker
docker_image  = ahlamloum/karmadock-seminar:v6
+WantGPUHomeMounted = true

requirements  = UidDomain == "cs.uni-saarland.de" && (Machine =!= "idun.hpc.uni-saarland.de")

executable    = scripts/run_infer.sh
arguments     = full_test full_scratch_karmadock_team002.pkl full_scratch

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = scripts, model/full_scratch_karmadock_team002.pkl, data/full_test.csv, data/full_test
transfer_output_files   = results

output = logs/full_test_infer.$(ClusterId).$(ProcId).out
error  = logs/full_test_infer.$(ClusterId).$(ProcId).err
log    = logs/full_test_infer.$(ClusterId).log

request_gpus   = 1
request_cpus   = 8
request_memory = 32GB

queue 1
```

تنقل مجموعة الاختبار الكاملة ذات 6,183 مركّبًا ونقطة تحقق البيانات الكاملة في المستودع، ثم تضع وسم الخرج `full_scratch`. تمرر نقطة التحقق كاسم مجرد لأن HTCondor ينقلها إلى الصندوق، ويحلها `run_infer.sh` مقابل `$PWD`. هذا استدلال على GPU واحدة لا DDP.

### `condor/full_train_scratch.sub` {#full-train-scratch-sub}

```condor
# full_train_scratch.sub  —  FULL-dataset training FROM SCRATCH (next phase, long job).
# Trains the paper's 2-stage protocol on full_train (23,483) / full_val (2,609) on one GPU.
# ... mounted-home rationale and prerequisite comments unchanged ...
universe      = docker
docker_image  = ahlamloum/karmadock-seminar:v6
+WantGPUHomeMounted = true
requirements  = UidDomain == "cs.uni-saarland.de" && (Machine =!= "idun.hpc.uni-saarland.de")

executable    = scripts/run_full_train.sh
arguments     = scratch
# Data/work default to $HOME/repro_test/{full_data,work_full}, resolved inside the container
# via the mounted home (portable across accounts). To relocate, uncomment and edit -- do NOT
# hardcode another user's home:
#   environment = "FULL_DATA_DIR=/path/to/full_data FULL_WORK_DIR=/path/to/work_full"

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = scripts
# data, graphs and checkpoints live on the mounted home -- nothing large is transferred

output = logs/full_train.$(ClusterId).$(ProcId).out
error  = logs/full_train.$(ClusterId).$(ProcId).err
log    = logs/full_train.$(ClusterId).log

request_gpus   = 1
request_cpus   = 8
request_memory = 64GB

queue 1
```

يستدعي غلاف البيانات الكاملة على GPU واحدة بوضع `scratch` الذي ينفذ المرحلتين. لا تنقل إلا الشيفرة؛ وتبقى البيانات ذات نحو 14 GB والرسوم ونقاط التحقق القابلة للاستئناف في المنزل المثبت. تطلب ذاكرة مضيف أكبر بكثير من الاستدلال لأن المعالجة المسبقة ومحملات رسوم التدريب أثقل.

### `condor/p1_baseline.sub` {#p1-baseline-sub}

```condor
# p1_baseline.sub  —  PIPELINE 1 (released weights) docking on proto_test.
# Produces the 3 pose variants (uncorrected / FF / align) in results/p1_baseline/.
# ... prerequisite comments unchanged ...
universe      = docker
docker_image  = ahlamloum/karmadock-seminar:v6
+WantGPUHomeMounted = true
requirements  = UidDomain == "cs.uni-saarland.de" && (Machine =!= "idun.hpc.uni-saarland.de")

executable    = scripts/run_infer.sh
arguments     = proto_test /app/KarmaDock/trained_models/karmadock_screening.pkl p1_baseline

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = scripts, data/proto_test.csv, data/prototype_model_data/proto_test
transfer_output_files   = results

output = logs/p1_baseline.$(ClusterId).$(ProcId).out
error  = logs/p1_baseline.$(ClusterId).$(ProcId).err
log    = logs/p1_baseline.$(ClusterId).log

request_gpus   = 1
request_cpus   = 8
request_memory = 32GB

queue 1
```

يمثل P1 استدلالًا فقط بنقطة التحقق المطلقة المنشورة داخل صورة Docker، فلا ينقل ملف نموذج. وتعكس بنية ملفاته وCSV وبرامجه ومجلد `results` المسارين P2 وP3 لضبط المقارنة.

### `condor/p2_scratch_infer.sub` {#p2-scratch-infer-sub}

```condor
# p2_scratch_infer.sub  —  PIPELINE 2 (from-scratch model) docking on proto_test.
# ... description and prerequisite comments unchanged ...
universe      = docker
docker_image  = ahlamloum/karmadock-seminar:v6
+WantGPUHomeMounted = true
requirements  = UidDomain == "cs.uni-saarland.de" && (Machine =!= "idun.hpc.uni-saarland.de")

executable    = scripts/run_infer.sh
arguments     = proto_test p2_scratch_karmadock_team002.pkl p2_scratch

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = scripts, model/p2_scratch_karmadock_team002.pkl, data/proto_test.csv, data/prototype_model_data/proto_test
transfer_output_files   = results

output = logs/p2_scratch.$(ClusterId).$(ProcId).out
error  = logs/p2_scratch.$(ClusterId).$(ProcId).err
log    = logs/p2_scratch.$(ClusterId).log

request_gpus   = 1
request_cpus   = 8
request_memory = 32GB

queue 1
```

يختلف P2 عن P1 في مصدر نقطة التحقق ووسمها فقط؛ إذ ينقل نموذج الفريق الأولي المدرّب من الصفر ويحل اسمه المجرد. يؤدي تثبيت البيانات والصورة والبذرة العشوائية داخل الغلاف والموارد إلى عزل أثر الأوزان.

### `condor/p2_train_scratch.sub` {#p2-train-scratch-sub}

```condor
# p2_train_scratch.sub  —  PIPELINE 2 training FROM SCRATCH (BONUS / long path).
# Preprocesses proto_train (712) then runs the paper's 2-stage protocol on one GPU.
# ... prerequisite comments unchanged ...
universe      = docker
docker_image  = ahlamloum/karmadock-seminar:v6
+WantGPUHomeMounted = true
requirements  = UidDomain == "cs.uni-saarland.de" && (Machine =!= "idun.hpc.uni-saarland.de")

executable    = scripts/run_train.sh
arguments     = scratch

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = scripts, data/prototype_model_data/proto_train.csv, data/prototype_model_data/proto_train
transfer_output_files   = ckpt

output = logs/p2_train.$(ClusterId).$(ProcId).out
error  = logs/p2_train.$(ClusterId).$(ProcId).err
log    = logs/p2_train.$(ClusterId).log

request_gpus   = 1
request_cpus   = 8
request_memory = 48GB

queue 1
```

تنقل مجموعة التدريب الأولية الصغيرة وتطلب `scratch`، فتجري المرحلة 1 الخاصة بـMDN فقط ثم المرحلة 2 المشتركة. لا يعاد إلا `ckpt`. ينشئ الغلاف أيضًا `work_train`، لكنه غير مدرج كمخرج؛ لذلك لا يفيد إلا داخل صندوق أُعيدت جدولته إذا حافظ الموقع على حالته، ولا تنقل الرسوم عند الإكمال العادي.

### `condor/p3_finetune.sub` {#p3-finetune-sub}

```condor
# p3_finetune.sub  —  PIPELINE 3 fine-tune training (BONUS / long path).
# ... description and prerequisite comments unchanged ...
universe      = docker
docker_image  = ahlamloum/karmadock-seminar:v6
+WantGPUHomeMounted = true
requirements  = UidDomain == "cs.uni-saarland.de" && (Machine =!= "idun.hpc.uni-saarland.de")

executable    = scripts/run_train.sh
arguments     = finetune

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = scripts, data/prototype_model_data/proto_train.csv, data/prototype_model_data/proto_train
transfer_output_files   = ckpt

output = logs/p3_train.$(ClusterId).$(ProcId).out
error  = logs/p3_train.$(ClusterId).$(ProcId).err
log    = logs/p3_train.$(ClusterId).log

request_gpus   = 1
request_cpus   = 8
request_memory = 48GB

queue 1
```

تطابق هذه المهمة تدريب P2 في الموارد والبيانات، لكنها تمرر `finetune`، فيبدأ الغلاف من أوزان الصورة المنشورة ويشغّل المرحلة ذات الخسارة المشتركة فقط. يجعل هذا التناظر فروق تدريب P2 وP3 راجعة إلى البروتوكول لا موارد الجدولة.

### `condor/p3_finetune_infer.sub` {#p3-finetune-infer-sub}

```condor
# p3_finetune_infer.sub  —  PIPELINE 3 (fine-tuned model) docking on proto_test.
# ... description and prerequisite comments unchanged ...
universe      = docker
docker_image  = ahlamloum/karmadock-seminar:v6
+WantGPUHomeMounted = true
requirements  = UidDomain == "cs.uni-saarland.de" && (Machine =!= "idun.hpc.uni-saarland.de")

executable    = scripts/run_infer.sh
arguments     = proto_test p3_finetune_karmadock_team002.pkl p3_finetune

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = scripts, model/p3_finetune_karmadock_team002.pkl, data/proto_test.csv, data/prototype_model_data/proto_test
transfer_output_files   = results

output = logs/p3_finetune.$(ClusterId).$(ProcId).out
error  = logs/p3_finetune.$(ClusterId).$(ProcId).err
log    = logs/p3_finetune.$(ClusterId).log

request_gpus   = 1
request_cpus   = 8
request_memory = 32GB

queue 1
```

يطابق استدلال P3 بنيويًا استدلال P2، لكنه ينقل نقطة التحقق المضبوطة ويختار وسم النتائج `p3_finetune`.

### `condor/posebusters_infer.sub` {#posebusters-infer-sub}

```condor
# posebusters_infer.sub  —  dock the 308-complex PoseBusters set with the FULL-DATA
# (multi-GPU Stage-2 trained) checkpoint. Produces the 3 pose variants
# (uncorrected / FF / align) in results/full_scratch/posebusters_filtered{,_ff,_align}/.
# ... prerequisite comments unchanged ...
universe      = docker
docker_image  = ahlamloum/karmadock-seminar:v6
+WantGPUHomeMounted = true
requirements  = UidDomain == "cs.uni-saarland.de" && (Machine =!= "idun.hpc.uni-saarland.de")

executable    = scripts/run_infer.sh
arguments     = posebusters_filtered full_scratch_karmadock_team002.pkl full_scratch

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = scripts, model/full_scratch_karmadock_team002.pkl, data/posebusters_filtered.csv, data/posebusters_filtered
transfer_output_files   = results

output = logs/posebusters_infer.$(ClusterId).$(ProcId).out
error  = logs/posebusters_infer.$(ClusterId).$(ProcId).err
log    = logs/posebusters_infer.$(ClusterId).log

request_gpus   = 1
request_cpus   = 8
request_memory = 32GB

queue 1
```

يعيد هذا استخدام نموذج البيانات الكاملة على معيار PoseBusters الأصغر. ملف الإرسال واضح داخليًا، لكن المحول المستدعى لا يتعرف حاليًا على مخطط `ligand_file` في CSV المرفق؛ راجع التنبيهات.

### `condor/smoke_2gpu.sub` {#smoke-2gpu-sub}

```condor
# smoke_2gpu.sub — SMOKE TEST of the DDP Stage-2 path: 2 GPUs, 3 epochs.
# Validates torchrun + NCCL + id-shard + all-reduce + rank-0 ckpt on real GPUs, and measures
# per-epoch wall time so the full run can be estimated. Isolated (writes ~/stage2_mgpu_test/work).
universe      = docker
docker_image  = ahlamloum/karmadock-seminar:v6
+WantGPUHomeMounted = true
requirements  = UidDomain == "cs.uni-saarland.de" && (Machine =!= "idun.hpc.uni-saarland.de")

executable    = run_full_stage2_ddp.sh
environment   = "S2_EPOCHS=3"

should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = train_ddp.py, seminar_csv.py, run_full_stage2_ddp.sh

output = logs/smoke.$(ClusterId).$(ProcId).out
error  = logs/smoke.$(ClusterId).$(ProcId).err
log    = logs/smoke.$(ClusterId).log

request_gpus   = 2
request_cpus   = 8
request_memory = 48GB

queue 1
```

يقصر اختبار الدخان التدريب على ثلاث حقب للتحقق من بدء العمليات واتصال NCCL وتقسيم المعرّفات وتخفيض المقاييس وحفظ نقطة التحقق على rank-0 قبل إنفاق أيام على التشغيل الكامل. وبما أنه يستخدم قيم المشغّل الأصلية (`BS=8` و`ACC=2`)، فإن دفعته الفعلية على وحدتي GPU هي 32 لا 64؛ فغرضه التحقق من المنظومة لا إنتاج تدريب مطابق للورقة.

## تنبيهات وملاحظات {#gotchas--notes}

- تُفسر مسارات الإرسال نسبة إلى المجلد الذي يُشغّل فيه `condor_submit`، لا بالضرورة مجلد ملف `.sub`. تتوقع ملفات النموذج الأولي الإرسال من جذر المستودع، بينما تتوقع ملفات DDP ذات الأسماء المجردة مجلد نشر يحتويها.
- يجب عمومًا أن يكون المجلد الأب `logs/` موجودًا لدى المرسل؛ لا تنشئه هذه الملفات.
- لا ينسخ `ON_EXIT` نقاط تحقق الصندوق باستمرار. تتجنب مهام البيانات الكاملة وDDP الخطر بالكتابة في المنزل المثبت، بينما تعتمد مهام النموذج الأولي على سلوك الصندوق وإعادة الجدولة حتى الخروج.
- يقيد `full_stage2_2gpu.sub` وحده قدرة GPU وذاكرتها. قد تطابق بقية مهام GPU أي بطاقة تحقق الطلب العام، بما فيها بطاقات مختلفة الذاكرة والأداء.
- ينقل `posebusters_infer.sub` الملف `data/posebusters_filtered.csv` ذي الأعمدة الفعلية `ligand_name,ligand_file,protein_file`. يستدعي `run_infer.sh` ‏`seminar_csv.complex_records` الذي لا يدعم هذا المخطط ويبحث بدلًا منه عن أعمدة البيانات الوصفية الكاملة. قد يفشل المسار كما هو بخطأ pandas من نوع `KeyError`؛ ويتعارض تعليق ملف الإرسال الذي يدعي قراءة `ligand_file` مع الشيفرة.
- لا تختصر الأسطر `# ... comments unchanged ...` أعلاه إلا تعليقات نثرية؛ أُعيد إنتاج كل توجيه تنفيذي من ملفات المصدر الـ12.
