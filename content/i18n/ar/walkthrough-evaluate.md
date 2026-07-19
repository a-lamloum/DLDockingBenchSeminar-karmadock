# `scripts/evaluate.sh` {#scriptsevaluatesh}

## الغرض {#purpose}

هذا هو مشغّل التقييم الدفعي للنموذج الأولي. يقيّم كل دليل نتائج متاح من حاصل ضرب P1/P2/P3 في الخام/FF/المحاذى باستخدام واجهة مجموعة `proto_test` نفسها، ثم ينسخ ملف CSV الخام لـP2 إلى اسم النتيجة الرئيس في المستودع.

## موقعه في المنظومة {#how-it-fits-in}

يستدعيه `condor/evaluate.sub` بعد مهام الاستدلال الثلاث للنموذج الأولي. ويتوقع `evaluation/evaluation.py` و`proto_test.csv` والبنى المرجعية وأشجار الوضعيات تحت `results/`. ويكتب `results/<pipeline>_<variant>_evaluation.csv`، ويمكن أن يكتب `results/proto_test_evaluation.csv`. لا توجد وسائط موضعية ولا متغيرات بيئة مخصصة.

## شرح الشيفرة {#walkthrough}

```bash
#!/usr/bin/env bash
# evaluate.sh — the scoring step (separate from docking).
# ...
set -euo pipefail

ROOT="$PWD"
EVALPY="$ROOT/evaluation/evaluation.py"
```

يجعل نمط Bash الصارم المسارات المفقودة وعمليات التقييم الفاشلة توقف المهمة. وبخلاف أغلفة التدريب والاستدلال، لا يستخدم `set -x`، لذلك لا تظهر إلا الرسائل الصريحة وسجلات المقيّم. يفترض البرنامج أن الدليل الحالي هو المستودع أو جذر المهمة المسطح.

```bash
first_existing () {   # <test-flag -f|-d> <candidate>...
    local flag=$1; shift
    local p; for p in "$@"; do [ "$flag" "$p" ] && { echo "$p"; return; }; done
}
CSV=$(first_existing -f "$ROOT/data/proto_test.csv" "$ROOT/proto_test.csv")
REF=$(first_existing -d "$ROOT/data/prototype_model_data/proto_test" "$ROOT/data/proto_test" "$ROOT/proto_test")

[ -n "$CSV" ]    || { echo "ERROR: proto_test.csv not found (looked in data/ and CWD)"; exit 1; }
[ -n "$REF" ]    || { echo "ERROR: proto_test reference dir not found — unzip data/prototype_model_data.zip -d data/prototype_model_data"; exit 1; }
[ -f "$EVALPY" ] || { echo "ERROR: missing $EVALPY"; exit 1; }
```

قد يحافظ HTCondor على أشجار الأدلة أو يسطّح العناصر المنقولة بحسب سياق التجهيز، لذلك تفحص `first_existing` عدة مواقع صالحة. وسيطها الأول عامل اختبار، `-f` للملف أو `-d` للدليل، وتترك `shift` المرشحات فقط. تحوّل الحراسات الثلاث نتيجة البحث الفارغة إلى رسالة خطأ قاتلة وواضحة.

```bash
eval_one () {   # <poses_dir> <out_csv>
    local poses=$1 out=$2
    local E; E="$(mktemp -d)"
    mkdir -p "$E/data" "$E/results"
    ln -sf "$CSV"   "$E/data/proto_test.csv"
    ln -sf "$REF"   "$E/data/proto_test"
    ln -sf "$poses" "$E/results/proto_test"
    ( cd "$E" && python3 "$EVALPY" --dataset proto_test --no_pb_valid --output_csv "$out" )
    rm -rf "$E"
}
```

لدى `evaluation.py` جذور نسبية ثابتة، `data/` و`results/`، ويشتق المسارات من اسم مجموعة البيانات. بدل نقل البيانات، تبني الدالة عرضًا مؤقتًا بروابط رمزية كي يظهر أي دليل وضعيات باسم `results/proto_test`. يقصر `--no_pb_valid` هذه المقارنة الدفعيّة عمدًا على RMSD ويتجنب فحوص PoseBusters المكلفة. ولا يغيّر أمر `cd` المحاط بقوسين دليل المستدعي.

```bash
for tag in p1_baseline p2_scratch p3_finetune; do
  for variant in ":uncorrected" "_ff:ff" "_align:align"; do
    suffix="${variant%%:*}"; label="${variant##*:}"
    poses="$ROOT/results/$tag/proto_test${suffix}"
    if [ ! -d "$poses" ]; then echo "skip $tag/$label (no poses at ${poses#$ROOT/})"; continue; fi
    eval_one "$poses" "$ROOT/results/${tag}_${label}_evaluation.csv"
    echo "wrote results/${tag}_${label}_evaluation.csv"
  done
done
```

تعرّف الحلقتان المتداخلتان مصفوفة 3×3. يُقسّم كل رمز `suffix:label` بتوسيع معاملات Bash؛ تزيل `%%:*` النقطتين وكل ما بعدهما، بينما تزيل `##*:` كل شيء حتى آخر نقطتين. تُتجاوز متغيرات الوضعيات المفقودة بدل إفشال المصفوفة كلها.

```bash
if [ -f "$ROOT/results/p2_scratch_uncorrected_evaluation.csv" ]; then
  cp "$ROOT/results/p2_scratch_uncorrected_evaluation.csv" "$ROOT/results/proto_test_evaluation.csv"
  echo "wrote results/proto_test_evaluation.csv (= P2 from-scratch, uncorrected)"
fi
```

تنشئ النسخة النهائية اسم النتيجة الافتراضي المتعارف عليه مع الاحتفاظ بملف CSV المصدر ذي الوسم الصريح.

## محاذير وملاحظات {#gotchas--notes}

- لا تُحذف الأدلة المؤقتة إلا في نهاية `eval_one` الناجحة؛ ولا يوجد trap، لذلك قد يترك التقييم المنقطع أو الفاشل دليلًا في منطقة النظام المؤقتة.
- يعطّل هذا الغلاف PoseBusters دائمًا مع أنه مفعّل افتراضيًا في مقيّم Python.
- يعرض الغلاف البيانات للمقيّم دائمًا باسم `proto_test` حتى عند المرور بمسارات نماذج مختلفة؛ وهذا صحيح لأن البنى واحدة ولا تتغير إلا التنبؤات.
- تستبدل إعادة التشغيل ملفات CSV الحالية للتقييم والنسخة الرئيسة.
