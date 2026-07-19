# النتائج {#results}

## 🚧 التسليم النهائي (البيانات الكاملة) — التقرير قيد الإعداد {#final-submission-full-data--report-in-progress}

تستخدم الجداول التالية مجموعة `proto_test` الأولية (136). يجري استكمال **التسليم النهائي للبيانات
الكاملة**: أُعيد تدريب نموذجنا من الصفر على تقسيم الندوة الكامل
(`model/full_scratch_karmadock_team002.pkl`؛ بروتوكول الورقة ذي المرحلتين 2-stage، مع المرحلة 2 على
2×A100 بسعة 40GB لكل بطاقة)، وقورن مباشرةً بأوزان المؤلفين المنشورة على مجموعتي `full_test`
(6,183) و`posebusters_filtered` (308) نفسيهما.

**النتيجة الأولية الرئيسة** (نجاح top-1 عند 2 Å باستخدام `evaluation.py` والوضعية الخام):

| المجموعة | نموذجنا (البيانات الكاملة) | الأوزان المنشورة |
|---|---|---|
| full_test (6,183) | 82.2 % | 88.3 % |
| PoseBusters (308) | 76.9 % (PB-Valid 5.2 %) | 83.1 % (PB-Valid 2.6 %) |

تتضمن الإضافة النموذج المدرّب (`model/`)، وملفات CSV الخاصة بمدخلات التقييم (`data/`)، وملفات
إرسال HPC في Condor (`condor/full_stage2_2gpu.sub` لتدريب المرحلة 2 على 2×A100، وملفا
`condor/{full_test,posebusters}_infer.sub` للاستدلال)، وسجلات المهام (`condor/logs/`). ما زال
التقرير الكامل ودفتر النتائج قيد الاستكمال.

---

## 3. النتائج {#3-results}

نموذج التسليم هو **P2 المدرّب من الصفر**، وهو المطلوب لإعادة تدريب الأداة على التقسيم المشترك.
يُعرض P1 المنشور وP3 المضبوط بدقة للسياق. أُجري التقييم بواسطة
[`evaluation/evaluation.py`](evaluation/evaluation.py) الرسمي على **136 مركّبًا في `proto_test`**
وعلى متغيرات المعالجة اللاحقة الثلاثة في KarmaDock.

**نسبة النجاح عند 2 Å (top-1):**

| المسار | الخام | مصحح FF | مصحح بالمحاذاة |
|---|---|---|---|
| **P2 — من الصفر** (نموذجنا) | **10.3 %** | 11.0 % | 94.1 % |
| P1 — خط الأساس المنشور | 80.9 % | 78.7 % | 95.6 % |
| P3 — الضبط الدقيق *(إضافي)* | 80.1 % | 75.0 % | 94.9 % |

القيم الخام **عند 1 Å / وسيط RMSD**: ‏P2 ‏3.7 % / 3.38 Å · ‏P1 ‏8.1 % / 1.45 Å · ‏P3 ‏7.4 % / 1.48 Å.
توجد ملفات CSV لكل مركّب في [`results/`](results): النمط `<pipeline>_<variant>_evaluation.csv`،
أما `proto_test_evaluation.csv` فهو النتيجة الرئيسة لـP2. النتائج حتمية مع `--random_seed 2023`.

> نعتقد أن نتيجة المحاذاة المصححة مضللة؛ إذ يُفترض وفق الورقة أن تكون أدنى من الوضعيتين الخام
> والمصححة بـFF، بينما حدث العكس في نتائجنا. يحتاج ذلك إلى تحقيق إضافي، ولا يؤثر في تدريب المرحلة التالية.

الوضعية **الخام** هي خرج النموذج مباشرة. متغير **FF** استرخاء باستخدام حقل قوة. أما متغير
**المحاذاة** فيضع الربيطة المتنبأ بها فوق الإطار المرجعي.

## تخطيط ملفات النتائج {#layout}

```
results/
├── proto_test/                         primary set = from-scratch P2, UNCORRECTED poses (136)
│                                        (what `python evaluation/evaluation.py --dataset proto_test` scores)
├── p1_baseline/  proto_test/  proto_test_ff/  proto_test_align/    136 each
├── p2_scratch/   proto_test/  proto_test_ff/  proto_test_align/    136 each
├── p3_finetune/  proto_test/  proto_test_ff/  proto_test_align/    136 each
├── proto_test_evaluation.csv           official RMSD for the primary set (= p2_scratch uncorrected)
├── <pipeline>_<variant>_evaluation.csv per-complex RMSD for each of the 9 pipeline x variant combos
└── README.md
```

ملفات الوضعيات هي `<complex_id>_pred.sdf`، وتُرتب المطابقات بدءًا من أفضل درجة MDN. المتغيرات
الثلاثة هي: الخام، واسترخاء حقل القوة FF، والمحاذاة إلى الإطار المرجعي.

## أعمدة ملف CSV للتقييم {#eval-csv-columns}

`complex_id, dataset, pose_rank, rmsd, rmsd_lt2, rmsd_lt1, ligand_file, protein_file`

يستخدم التقييم `GetBestRMS` في RDKit مع تصحيح التناظر وأفضل وضعية top-1. تساوي `success@2Å`
نسبة الصفوف التي تحقق `pose_rank == 1` و`rmsd_lt2 == True`.

## تنزيل تقييمات كل مركّب {#per-complex-evaluation-downloads}

- [`p1_baseline_align_evaluation.csv`](results/p1_baseline_align_evaluation.csv)
- [`p1_baseline_ff_evaluation.csv`](results/p1_baseline_ff_evaluation.csv)
- [`p1_baseline_uncorrected_evaluation.csv`](results/p1_baseline_uncorrected_evaluation.csv)
- [`p2_scratch_align_evaluation.csv`](results/p2_scratch_align_evaluation.csv)
- [`p2_scratch_ff_evaluation.csv`](results/p2_scratch_ff_evaluation.csv)
- [`p2_scratch_uncorrected_evaluation.csv`](results/p2_scratch_uncorrected_evaluation.csv)
- [`p3_finetune_align_evaluation.csv`](results/p3_finetune_align_evaluation.csv)
- [`p3_finetune_ff_evaluation.csv`](results/p3_finetune_ff_evaluation.csv)
- [`p3_finetune_uncorrected_evaluation.csv`](results/p3_finetune_uncorrected_evaluation.csv)
- [`proto_test_evaluation.csv`](results/proto_test_evaluation.csv)
