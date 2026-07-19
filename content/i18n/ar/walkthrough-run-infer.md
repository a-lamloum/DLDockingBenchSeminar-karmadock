# شرح `scripts/run_infer.sh` {#scriptsruninfersh}

## الغرض {#purpose}

يشغّل هذا البرنامج مسار توليد الوضعيات الكامل لزوج واحد من مجموعة البيانات ونقطة التحقق. يحوّل البنى، وينشئ الجيوب والرسوم البيانية، ويستدعي إرساء KarmaDock الأصلي، ويصدّر الوضعيات الخام والمصغرة بحقل القوة والمصححة بالمحاذاة بتنسيق الندوة.

## موقعه في المنظومة {#how-it-fits-in}

تستدعيه ملفات إرسال الاستدلال الخمسة المدرجة بثلاث وسائط موضعية: اسم المجموعة، ومسار أو اسم نقطة التحقق، ووسم الخرج. يقرأ `<dataset>.csv` و`<dataset>/` من صندوق المهمة الحالي، وينشئ بيانات وسيطة مؤقتة، ويكتب `results/<tag>/<dataset>{,_ff,_align}/<id>_pred.sdf`. يحدث التقييم لاحقًا في `evaluate.sh` أو `evaluation.py`.

## شرح الشيفرة {#walkthrough}

```bash
#!/usr/bin/env bash
# run_infer.sh <dataset> <model_file> <tag>
# ...
set -euo pipefail
set -x

DATASET=$1
MODEL=$2
TAG=$3
```

يوقف الوضع الصارم المسارات الجزئية، ويجعل التتبع سجل العنقود قابلًا لإعادة الإنتاج. بخلاف `run_train.sh`، لا تستخدم هذه المتغيرات `${1:?...}`؛ ومع `set -u` يؤدي غياب أي منها إلى إخفاق فوري، لكن بلا رسالة استخدام مخصصة.

```bash
SUBMIT="$PWD"
case "$MODEL" in /*) : ;; *) MODEL="$SUBMIT/$MODEL" ;; esac

KD=/app/KarmaDock
WORK="$(mktemp -d -t kd-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
KIN="$WORK/kin"; GRAPH="$WORK/graphs"; KDOUT="$WORK/kdout"
mkdir -p "$KIN" "$GRAPH" "$KDOUT"
```

يميز `case` المسارات المطلقة `/*` من أسماء نقاط التحقق المنقولة. تُحل النقاط النسبية قبل تغيير المجلد لاحقًا. يعزل `mktemp -d` التشغيل، ويحذف `trap` المقتبس المساحة عند الخروج الطبيعي أو معظم الأخطاء. لا تبقى إلا النتائج المصدرة.

```bash
echo "=== [$TAG] dock $DATASET with $MODEL ==="

python3 "$SUBMIT/scripts/convert_seminar_to_karmadock.py" \
    --csv "$SUBMIT/$DATASET.csv" --src_dir "$SUBMIT/$DATASET" --out_dir "$KIN"

cd "$KD/utils"
python3 -u pre_processing.py  --complex_file_dir "$KIN"
python3 -u generate_graph.py  --complex_file_dir "$KIN" --graph_file_dir "$GRAPH"
echo "=== [$TAG] graphs built: $(ls "$GRAPH" | wc -l) ==="
```

يغير الأمر الأول التسمية والتخطيط فقط. تحدد المعالجة الأصلية الجيب، ثم يحول توليد الرسم الذرات والبقايا وعلاقاتها إلى موترات رسوم DGL. العدد فحص سريع للتغطية، وليس تحققًا من نجاح كل معرّف متوقع.

```bash
python3 -u ligand_docking.py \
    --graph_file_dir "$GRAPH" \
    --model_file "$MODEL" \
    --out_dir "$KDOUT" \
    --docking True --scoring True --correct True \
    --batch_size 64 --random_seed 2023
```

هذه نقطة دخول الاستدلال الأصلية في KarmaDock. تتنبأ `--docking True` بالإحداثيات، وتقيّم `--scoring True` الوضعيات برأس MDN، وتطلب `--correct True` الأشكال الثلاثة. الخرج الخام غير مصحح؛ ويجري تصحيح FF استرخاءً بالميكانيكا الجزيئية، بينما تضع المحاذاة الإحداثيات فوق إطار مرجعي. تجعل البذرة الثابتة الوضعيات الابتدائية والتكرارات العشوائية قابلة لإعادة الإنتاج ضمن بيئة البرامج والعتاد نفسها.

```bash
export_variant () {
    local mode=$1 suffix=$2
    local out="$SUBMIT/results/$TAG/${DATASET}${suffix}"
    mkdir -p "$out"
    python3 "$SUBMIT/scripts/convert_karmadock_to_seminar.py" \
        --input_dir "$KDOUT" --csv "$SUBMIT/$DATASET.csv" --out_dir "$out" --mode "$mode"
    echo "=== [$TAG] ${mode}: $(ls "$out"/*_pred.sdf 2>/dev/null | wc -l) poses -> ${out#$SUBMIT/} ==="
}
export_variant uncorrected     ""
export_variant ff_corrected    "_ff"
export_variant align_corrected "_align"

echo "=== [$TAG] inference done (3 pose variants in results/$TAG/) ==="
```

تجمع الدالة استدعاءات المحول المتكررة. تزيل `${out#$SUBMIT/}` بادئة مجلد الإرسال للتسجيل الأنظف فقط. يرتب المحول تكرارات الإرساء وفق الدرجة، ولذلك تكون سجلات SDF الناتجة مرتبة من الأفضل. تنشئ الاستدعاءات الثلاثة أشجار نتائج متوازية لمقارنة المعالجة اللاحقة بعدل.

## تنبيهات وملاحظات {#gotchas--notes}

- تُحذف الرسوم المؤقتة وخرج KarmaDock الأصلي دائمًا عند الخروج، ولذلك يتطلب تصحيحها تعديل البرنامج أو نسخها قبل الخروج.
- تستخدم أوامر عد الرسوم والوضعيات خطوط أنابيب مع `pipefail`؛ يخفي `2>/dev/null` تشخيص عدم مطابقة النمط، لكن مجلد الوضعيات الفارغ قد يجعل استبدال الأمر يفشل.
- يدعم `seminar_csv.py` الحالي `ligand_file_name` أو أعمدة البيانات الوصفية الكاملة الأربعة، بينما يستخدم `data/posebusters_filtered.csv` المرفق `ligand_file`. لذلك قد يرفع مسار إرسال PoseBusters خطأ عمود مفقود قبل الاستدلال.
- تستخدم المحاذاة الوضعية أو الإطار المرجعي، ولذلك ينبغي تفسيرها كتشخيص معالجة لاحقة لا كخرج إرساء أعمى مستقل عن الإجابة.
