# `scripts/run_train.sh` {#scriptsruntrainsh}

## الغرض {#purpose}

هذا مشغّل تدريب النموذج الأولي المكتفي بذاته. يحوّل `proto_train` ويعالجه مسبقًا، ثم يشغّل إما بروتوكول التدريب من الصفر ذي المرحلتين، أي P2، أو يضبط نقطة التحقق المنشورة ضبطًا دقيقًا، أي P3، باستخدام حلقة `train.py` ذات العملية الواحدة.

## موقعه في المنظومة {#how-it-fits-in}

يستدعيه `condor/p2_train_scratch.sub` بالوسيط `scratch`، ويستدعيه `condor/p3_finetune.sub` بالوسيط `finetune`. ويتوقع `proto_train.csv` و`proto_train/` و`scripts/` في صندوق HTCondor الحالي، إضافة إلى KarmaDock الأصلي في `/app/KarmaDock`. ويكتب معالجة مسبقة قابلة لإعادة الاستخدام تحت `work_train/`، ونقاط التحقق والسجلات تحت `ckpt/`.

## شرح الشيفرة {#walkthrough}

```bash
#!/usr/bin/env bash
# run_train.sh <scratch|finetune>
# ...
set -euo pipefail
set -x

MODE=${1:?usage: run_train.sh <scratch|finetune>}
SUBMIT="$PWD"
KD=/app/KarmaDock
export PYTHONPATH=/app/KarmaDock:${PYTHONPATH:-}
```

يعثر سطر `env` الافتتاحي على Bash عبر البيئة. يوقف `set -e` التنفيذ عند أمر فاشل غير معالج، ويرفض `-u` المتغيرات غير المضبوطة، ويجعل `pipefail` خط الأنابيب يفشل إذا فشل أي مكوّن. ويطبع `set -x` الأوامر الموسعة لتصحيح أخطاء العنقود. يجعل `${1:?...}` الوسيط الموضعي الأول إلزاميًا. وتبقى `${PYTHONPATH:-}` آمنة تحت `-u` وتتيح استيرادات KarmaDock الأصلية لبرامج Python المحلية.

```bash
# Persistent work dir (NOT mktemp) so --resume survives a force-reschedule.
WORK="$SUBMIT/work_train"; KIN="$WORK/complex"; GRAPH="$WORK/graphs"
mkdir -p "$KIN" "$GRAPH" "$SUBMIT/ckpt"

if [ ! -f "$WORK/.preprocessed" ]; then
  python3 "$SUBMIT/scripts/convert_seminar_to_karmadock.py" \
      --csv "$SUBMIT/proto_train.csv" --src_dir "$SUBMIT/proto_train" --out_dir "$KIN"
  ( cd "$KD/utils"
    python3 -u pre_processing.py --complex_file_dir "$KIN"
    python3 -u generate_graph.py --complex_file_dir "$KIN" --graph_file_dir "$GRAPH" )
  touch "$WORK/.preprocessed"
fi
```

دليل العمل دائم كي تستطيع مهمة عنقودية أُخرجت أو أعيدت جدولتها استخدام الرسوم ونقاط تحقق `--resume`. ينشئ المحوّل تخطيط مدخلات KarmaDock؛ وتستخرج `pre_processing.py` الأصلية الجيوب وتطبّعها، بينما تسلسل `generate_graph.py` كائنات الرسوم في ملفات `.dgl`. تجعل الأقواس أمر `cd` محليًا لعملية فرعية. ويمنح خيار Python ‏`-u` سجلات غير مخزنة مؤقتًا. لا تُلمس العلامة إلا إذا نجحت أوامر المعالجة كلها تحت `set -e`.

```bash
cd "$SUBMIT"
if [ "$MODE" = "scratch" ]; then
  if [ ! -f ckpt/p2_stage1/stage.done ]; then
    python3 -u scripts/train.py --csv proto_train.csv --graph_dir "$GRAPH" --complex_dir "$KIN" \
        --out_dir ckpt/p2_stage1 --init_model "" --pos_r 0 --lr 1e-3 --weight_decay 1e-5 \
        --batch_size 4 --accum_steps 16 --patience 70 --epochs 1000 --val_frac 0.1 --random_seed 42 --resume
    touch ckpt/p2_stage1/stage.done
  fi
```

تبدأ Stage 1 بأوزان عشوائية، أي `--init_model ""`، وتضبط `pos_r=0`، فتُضرب خسارة الإحداثيات/RMSD في صفر ويركز التدريب على هدف تسجيل MDN. ينفذ **تراكم التدرجات** تحديثًا واحدًا للمحسّن بعد 16 دفعة فيزيائية حجم كل منها 4، بما يقارب دفعة فعّالة مقدارها 64 دون تخزين الأمثلة الـ64 معًا. وقد يوقف الإيقاف المبكر الحقب الاسمية الـ1000 بعد 70 عملية تحقق بلا تحسن. وتمنع `stage.done` إعادة Stage 1 المكتملة.

```bash
  python3 -u scripts/train.py --csv proto_train.csv --graph_dir "$GRAPH" --complex_dir "$KIN" \
      --out_dir ckpt/p2_stage2 --init_model ckpt/p2_stage1/karmadock_team002.pkl --pos_r 1 \
      --lr 1e-4 --weight_decay 1e-4 --batch_size 4 --accum_steps 16 --patience 20 --epochs 1000 \
      --val_frac 0.1 --random_seed 42 --jitter 0.05 --resume
  echo "=== P2 from-scratch done: ckpt/p2_stage2/karmadock_team002.pkl ==="
```

تبدأ Stage 2 من أفضل نقطة تحقق لـStage-1 وتفعّل تعلم الإحداثيات باستخدام `pos_r=1`. وتستخدم معدل تعلم أصغر واضمحلال أوزان أقوى وpatience أقصر وتشويش إحداثيات غاوسيًا مقداره 0.05 Å لزيادة البيانات. تعيد عملية تنقيح الوضعية في KarmaDock تدوير شبكة تحديث الإحداثيات المتكافئة ثلاث مرات داخل النموذج الأصلي؛ ولا يطبق هذا البرنامج تلك البنية، لكن Stage 2 توفّر إشارة RMSD التي تدربها.

```bash
elif [ "$MODE" = "finetune" ]; then
  python3 -u scripts/train.py --csv proto_train.csv --graph_dir "$GRAPH" --complex_dir "$KIN" \
      --out_dir ckpt/p3_finetune --init_model /app/KarmaDock/trained_models/karmadock_screening.pkl \
      --pos_r 1 --lr 1e-4 --weight_decay 0 --batch_size 4 --accum_steps 16 --patience 30 --epochs 500 \
      --val_frac 0.1 --random_seed 42 --resume
  echo "=== P3 fine-tune done: ckpt/p3_finetune/karmadock_team002.pkl ==="
else
  echo "ERROR: mode must be 'scratch' or 'finetune'"; exit 2
fi
```

يتجاوز الضبط الدقيق Stage 1 ويبدأ بأوزان الفحص المنشورة، مع تحسين الخسارتين. ويرفض الفرع الأخير أي قيمة غير النمطين المعلنين بحالة خروج 2.

## محاذير وملاحظات {#gotchas--notes}

- تؤكد علامتا `.preprocessed` و`stage.done` الاكتمال من دون فحص ما إذا كانت الملفات حُذفت أو تلفت لاحقًا؛ احذف العلامة لفرض إعادة التوليد.
- لا يستأنف `--resume` إلا إذا وُجد `last.pt`. ولا تملك Stage 2 المكتملة علامة مستقلة، لذلك تُستدعى مجددًا، مع إمكان استئناف نقطة تحققها قرب النهاية السابقة.
- تكون عبارة «الدفعة الفعّالة 64» دقيقة لنوافذ التراكم الكاملة؛ أما النافذة الجزئية الأخيرة فتظل مقسومة على 16 في `train.py`، لذلك يكون تدرجها أصغر.
