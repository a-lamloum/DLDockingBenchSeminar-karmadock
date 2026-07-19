# `scripts/run_full_train.sh` {#scriptsrunfulltrainsh}

## الغرض {#purpose}

هذا نظير `run_train.sh` للبيانات الكاملة على GPU واحدة. ولأن البنى الكاملة والرسوم المولّدة كبيرة، فإنه يقرأها ويكتبها عبر دليل المنزل المركّب داخل مهمة Docker بدل نقلها مع كل إرسال إلى HTCondor.

## موقعه في المنظومة {#how-it-fits-in}

يستدعيه `condor/full_train_scratch.sub` بالنمط `scratch`؛ كما يطبق `finetune` مع أن ملفات الإرسال المذكورة لا تستخدم ذلك النمط. يقرأ `$FULL_DATA_DIR/{full_train,full_val}.csv` وأدلة البنى المطابقة، مع القيمة الافتراضية `$HOME/repro_test/full_data`. ويحفظ البيانات المحوّلة والرسوم وحالة W&B ونقاط التحقق تحت `$FULL_WORK_DIR`، وقيمته الافتراضية `$HOME/repro_test/work_full`.

## شرح الشيفرة {#walkthrough}

```bash
#!/usr/bin/env bash
# run_full_train.sh <scratch|finetune>
# ...
set -euo pipefail
set -x

MODE=${1:?usage: run_full_train.sh <scratch|finetune>}
KD=/app/KarmaDock
export PYTHONPATH=/app/KarmaDock:${PYTHONPATH:-}

DATA="${FULL_DATA_DIR:-$HOME/repro_test/full_data}"
WORK="${FULL_WORK_DIR:-$HOME/repro_test/work_full}"
SUBMIT="$PWD"; SCRIPTS="$SUBMIT/scripts"
CKPT="$WORK/ckpt"; mkdir -p "$CKPT"
```

لخيارات Bash الصارمة وخيار التصحيح الأدوار نفسها التي في `run_train.sh`. وتجعل `${VAR:-default}` المسارات قابلة للنقل. يجهّز HTCondor دليل `scripts` تحت دليل عمل الصندوق مع أن الملف التنفيذي نفسه قد يظهر مسطحًا، لذلك تستخدم الشيفرة عمدًا `$PWD/scripts` بدل دليل الملف التنفيذي.

```bash
preprocess() {                               # $1 = split name (full_train | full_val)
  local split="$1"
  local kin="$WORK/$split/complex" graph="$WORK/$split/graphs"
  if [ -f "$WORK/$split/.preprocessed" ]; then echo "# $split already preprocessed"; return; fi
  mkdir -p "$kin" "$graph"
  python3 "$SCRIPTS/convert_seminar_to_karmadock.py" \
      --csv "$DATA/$split.csv" --src_dir "$DATA/$split" --out_dir "$kin"
  ( cd "$KD/utils"
    python3 -u pre_processing.py --complex_file_dir "$kin"
    python3 -u generate_graph.py --complex_file_dir "$kin" --graph_file_dir "$graph" )
  touch "$WORK/$split/.preprocessed"
}
preprocess full_train
preprocess full_val
```

تزيل الدالة التكرار بين تقسيمي التدريب والتحقق المنتقيين. وتحصر `local` متغيراتها كي لا تتسرب إلى حالة البرنامج العامة. يحصل كل تقسيم على بنى محوّلة ودليل رسوم مستقلين. وتهم رسوم التحقق المنفصلة لأن التحقق الكامل منتقى بدل اقتطاعه عشوائيًا من التدريب.

```bash
COMMON=( --csv "$DATA/full_train.csv" --graph_dir "$WORK/full_train/graphs"
         --complex_dir "$WORK/full_train/complex"
         --val_csv "$DATA/full_val.csv" --val_graph_dir "$WORK/full_val/graphs"
         --batch_size 4 --accum_steps 16 --random_seed 42 --resume
         --wandb --wandb_project karmadock-seminar --wandb_run_name "full_${MODE}" )
```

تحافظ مصفوفة Bash على كل وسيط بوصفه كلمة Shell مستقلة حتى إذا احتوى المسار مسافات. تشترك الأنماط كلها في تقسيم التحقق الصريح المقاوم للتسرب، وإعدادات الدفعة، وسلوك الاستئناف، وتسجيل W&B. تُتوقع بيانات اعتماد W&B عبر `WANDB_API_KEY` أو ملف `.netrc` في المنزل المركّب؛ وإلا يعود `train.py` إلى التسجيل دون اتصال.

```bash
if [ "$MODE" = "scratch" ]; then
  if [ ! -f "$CKPT/p_stage1/stage.done" ]; then
    python3 -u "$SCRIPTS/train.py" "${COMMON[@]}" \
        --out_dir "$CKPT/p_stage1" --init_model "" --pos_r 0 \
        --lr 1e-3 --weight_decay 1e-5 --patience 70 --epochs 1000
    touch "$CKPT/p_stage1/stage.done"
  fi
  python3 -u "$SCRIPTS/train.py" "${COMMON[@]}" \
      --out_dir "$CKPT/p_stage2" --init_model "$CKPT/p_stage1/karmadock_team002.pkl" \
      --pos_r 1 --lr 1e-4 --weight_decay 1e-4 --patience 20 --epochs 1000 --jitter 0.05
  echo "=== FULL from-scratch done: $CKPT/p_stage2/karmadock_team002.pkl ==="
```

هذا هو الهدف ذو المرحلتين نفسه في مشغّل النموذج الأولي: تسجيل MDN فقط أولًا، ثم تعلّم التسجيل وإحداثيات الإرساء معًا. يعني CSV الصريح للتحقق أن كل رسوم `full_train` المتاحة تظل للتدريب وكل رسوم `full_val` المتاحة تظل للتحقق.

```bash
elif [ "$MODE" = "finetune" ]; then
  python3 -u "$SCRIPTS/train.py" "${COMMON[@]}" \
      --out_dir "$CKPT/p_finetune" --init_model "$KD/trained_models/karmadock_screening.pkl" \
      --pos_r 1 --lr 1e-4 --weight_decay 0 --patience 30 --epochs 500
  echo "=== FULL fine-tune done: $CKPT/p_finetune/karmadock_team002.pkl ==="
else
  echo "ERROR: mode must be 'scratch' or 'finetune'"; exit 2
fi
```

يبدأ البديل من الأوزان المنشورة ويدرّب مرحلة واحدة بخسارة مشتركة. يأتي فحص النمط متأخرًا عمدًا لأن المعالجة المسبقة تسبقه؛ وتمرر ملفات الإرسال الصالحة نمطًا معروفًا دائمًا.

## محاذير وملاحظات {#gotchas--notes}

- يفترض البرنامج أن HTCondor ركّب منزل المستخدم، أي `+WantGPUHomeMounted = true` في ملف الإرسال. وبدونه قد تشير القيم الافتراضية إلى مسارات غير متاحة في الحاوية.
- تحمل علامات المعالجة المسبقة خطر العلامة القديمة نفسه الموجود في `run_train.sh`.
- يحدث التحقق من النمط بعد معالجة التقسيمين، لذلك قد ينفذ نمط غير صالح معالجة مسبقة مكلفة قبل الخروج.
- تصف التعليقات التدريب الكامل على GPU واحدة بأنه غير عملي صراحة؛ فهذا المشغّل يعمل لكنه لا يستفيد من عدة وحدات GPU.
