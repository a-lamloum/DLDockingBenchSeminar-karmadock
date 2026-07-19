# 4. التقييم وإعادة الإنتاج {#4-evaluate--reproduce}

تتضمن المستودعات الوضعيات المتنبأ بها (مخزنة **من دون ضغط**) وملفات CSV الناتجة من
[`evaluation.py`](evaluation/evaluation.py) الرسمي. يمكن قراءة النتائج مباشرة، أو إعادة تشغيل
المقيّم محليًا على الوضعيات المرفقة من دون الحاجة إلى العنقود.

### A. تقييم الوضعيات المرفقة {#a-score-the-shipped-poses}

```bash
# one-time setup: unzip the reference structures and let evaluation.py find them
unzip -o data/prototype_model_data.zip -d data/prototype_model_data
ln -sf prototype_model_data/proto_test data/proto_test      # data/proto_test.csv already ships

# score results/proto_test/ (= our from-scratch P2 model) -> results/proto_test_evaluation.csv
python evaluation/evaluation.py --dataset proto_test
```

يضم `results/proto_test/` وضعيات نموذج **P2 المدرّب من الصفر**، وهو نموذج التسليم الرئيس. لتقييم
مسار أو متغير آخر، اجعل `results/proto_test/` يشير إليه، مثل
`rm results/proto_test && ln -s p1_baseline/proto_test results/proto_test`، ثم أعد التشغيل. توجد
ملفات CSV المحسوبة مسبقًا لكل مسار × متغير في [`results/`](results).

### B. إعادة توليد الوضعيات على العنقود (اختياري) {#b-regenerate-the-poses-on-the-cluster-optional}

```bash
mkdir -p logs
# dock: each job preprocesses, docks (seed 2023) and exports the 3 pose variants
condor_submit condor/p1_baseline.sub        # -> results/p1_baseline/{proto_test,_ff,_align}
condor_submit condor/p2_scratch_infer.sub   # -> results/p2_scratch/...
condor_submit condor/p3_finetune_infer.sub  # -> results/p3_finetune/...
# score (after the 3 docking jobs finish): runs evaluation.py over every pipeline x variant
condor_submit condor/evaluate.sub
```

ملفات الإرسال في [`condor/`](condor) **قابلة للنقل**: تنقل البيانات وشيفرتنا إلى المهمة، وتستخدم
الأوزان المنشورة المضمّنة في الصورة، فلا تحتاج إلى تعديل مسارات. الإرساء حتمي مع
`--random_seed 2023`.

> مجموعة التقييم هي [`data/proto_test.csv`](data/proto_test.csv) وتضم 136 مركّبًا. تأتي البنى
> المرجعية من الحزمة، بينما تبقى `proto_train` التي تضم 712 مركّبًا بلا تغيير.

### إعادة التدريب من الصفر والضبط الدقيق {#retraining-from-scratch-and-fine-tuning}

توجد المعاملات الفائقة في [القسم 5](#5-training-information--parameters-from-the-paper). مشغلا
العنقود هما [`condor/p2_train_scratch.sub`](condor/p2_train_scratch.sub) للمسار P2 و
[`condor/p3_finetune.sub`](condor/p3_finetune.sub) للمسار P3. التدريب هو المسار الطويل: يستغرق
نحو **~54 hours** على GPU واحد لمعالجة 712 مركّبًا في proto_train.
