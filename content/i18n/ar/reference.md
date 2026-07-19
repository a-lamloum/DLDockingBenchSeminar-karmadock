# المرجع {#reference}

## 6. تخطيط المستودع {#6-repository-layout}

| المسار | الوصف |
|---|---|
| [`README.md`](README.md) | ملف العرض الرئيس |
| [`Dockerfile`](Dockerfile) | صورة `ahlamloum/karmadock-seminar:v6` |
| [`scripts/`](scripts) | `train.py`، وهو الأثر الرئيس، والمحولات، و`run_infer.sh`، و`evaluate.sh`، و`run_train.sh` |
| [`condor/`](condor) | ملفات إرسال HTCondor قابلة للنقل: 3 للإرساء + 1 للتقييم + 2 للتدريب |
| [`evaluation/evaluation.py`](evaluation/evaluation.py) | مقيّم الندوة الرسمي من دون تعديل |
| [`model/`](model) | نقاط تحقق P2 وP3 المدرّبة، بحجم يقارب ~15 MB لكل منها |
| [`notebooks/results_and_comparison.ipynb`](notebooks/results_and_comparison.ipynb) | الجداول والرسوم |
| [`results/`](results) | `proto_test/` = وضعيات P2 التي يقيّمها `evaluation.py --dataset proto_test`؛ و`<pipeline>/proto_test{,_ff,_align}/` = المسارات 3 × المتغيرات 3؛ و`*_evaluation.csv` = RMSD الرسمي لكل مركّب ومتغير |
| [`docs/`](docs) | سجلات التدريب والأشكال |
| [`data/proto_test.csv`](data/proto_test.csv) | تعيين proto_test الذي يضم 136 مركّبًا |
| [`data/prototype_model_data.zip`](data/prototype_model_data.zip) | البنى المرجعية: proto_test وproto_train وملفات SDF/PDB المنقحة |
| [`scripts/README.md`](scripts/README.md) | مصدر الشيفرة: عملنا مقارنةً بـKarmaDock الأصلي |

**صورة Docker:** ‏`ahlamloum/karmadock-seminar:v6` على Docker Hub.

## 7. المشكلات والإصلاحات {#7-issues--fixes}

- **مشكلة العقدة:** كانت عقد GPU في عنقود idun تعيد خطأ أحيانًا أثناء التشغيل، لذلك استُبعدت
  العقدة المحددة في متطلبات ملفات الإرسال. الإصلاح:
  `requirements = … && (Machine =!= "idun.hpc.uni-saarland.de")`. **[solved]**.
- **قيود الموارد:** بقيت مهام `request_gpus=2/4` خاملة أيامًا. يعمل النموذج الأولي على GPU واحد؛
  تعذر تشغيله على 2 GPU، لذلك غُيّر الطلب إلى 1 GPU. لم يُحسم ما إذا كان ذلك ملائمًا للبيانات
  الكاملة بسبب طول زمن التشغيل. الإصلاح: طلب 1 GPU. **[solved]**.

## التأليف ومصدر الشيفرة {#authorship--provenance}

**النقطة الأساسية:** يوفّر KarmaDock العام (`schrojunzhang/KarmaDock`) شيفرة الاستدلال والأوزان
المدرّبة مسبقًا فقط، **ولا يوفّر برنامج تدريب**. مساهمتنا في الندوة هي حلقة التدريب، ومحولات
البيانات، وأغلفة التشغيل، ومنظومة HTCondor الكاملة المبنية حول KarmaDock. نستدعي وحدات KarmaDock
كما هي، بما يشمل النموذج والمعالجة المسبقة والإرساء، ولم نعدّل أي ملف داخل `KarmaDock/` الأصلي؛
يستنسخه `Dockerfile` من جديد، وتوجد شيفرتنا كلها في `scripts/`.

### الملفات التي أنشأناها {#files-we-created}

### `scripts/` {#scripts}

| الملف | وظيفته | أجزاء KarmaDock التي يستدعيها |
|---|---|---|
| `train.py` | **أثرنا الرئيس**: حلقة كاملة للتدريب والضبط الدقيق مع نقاط تحقق، ومُحسّن، وإيقاف مبكر، وتجميع تدرجات، وتقسيم تحقق، وW&B، و`--resume` | يستورد نموذج `KarmaDock` الأصلي، و`PDBBindGraphDataset`، و`PassNoneDataLoader`، و`set_random_seed`/`Early_stopper`؛ ويستخدم خسائر `forward()` الخاصة بالنموذج |
| `convert_seminar_to_karmadock.py` | يحول تخطيط بيانات الندوة إلى تخطيط KarmaDock | — |
| `convert_karmadock_to_seminar.py` | يحول الوضعيات المرساة إلى `<id>_pred.sdf` في تنسيق الندوة، بدءًا من الأفضل | يقرأ ملفات SDF للوضعيات باستخدام RDKit |
| `run_infer.sh` | إرساء قابل للنقل: معالجة مسبقة، ثم إرساء، ثم تصدير المتغيرات 3: الخام وFF والمحاذاة | يستدعي `pre_processing.py` و`generate_graph.py` و`ligand_docking.py` الأصليين، إلى جانب محولاتنا |
| `evaluate.sh` | خطوة تقييم منفصلة تشغّل `evaluation.py` لكل مسار × متغير | تستدعي `evaluation/evaluation.py` الرسمي |
| `run_train.sh` | تدريب قابل للنقل: `scratch` = المسار P2 ذو المرحلتين (2-stage)، و`finetune` = المسار P3 من الأوزان المنشورة؛ يعالج `proto_train` مسبقًا | يستدعي `train.py` الخاص بنا |

### `condor/` (عملنا) {#condor-our-work}

| الملف | المهمة |
|---|---|
| `p1_baseline.sub` | استدلال P1 بالأوزان المنشورة + تقييم |
| `p2_scratch_infer.sub` | استدلال P2 بنقطة التحقق المدرّبة من الصفر + تقييم |
| `p3_finetune_infer.sub` | استدلال P3 بنقطة التحقق المضبوطة دقيقًا + تقييم |
| `p2_train_scratch.sub` | تدريب P2 من الصفر على مرحلتين (2-stage) |
| `p3_finetune.sub` | تدريب الضبط الدقيق P3 |
| `evaluate.sub` | تشغيل البرنامج النصي evaluate.sh |

### أعمال أخرى لنا {#other-ours}

- يبني `Dockerfile` الصورة `ahlamloum/karmadock-seminar:v6`؛ إذ يستنسخ KarmaDock، ويثبت بيئة
  conda المحزّمة من المؤلفين، ويضيف `scripts/` الخاصة بنا.
- نقاط التحقق في `model/` للمسارين P2 وP3، ودفتر النتائج، وكل الوثائق، وسجلات التدريب في `docs/`.

## الخلاصة {#summary}

- **عمل المؤلفين:** نموذج KarmaDock، وأدوات المعالجة المسبقة والإرساء، والأوزان المنشورة؛ استُخدمت كما هي.
- **عملنا:** `train.py`، إذ لم يكن هناك مدرّب، ومحولات البيانات، وأغلفة التشغيل، وكل ملف `.sub`
  في Condor، وبناء Docker، وكل التحليلات والوثائق. **لم يُعدّل أي ملف مصدر في KarmaDock.**
