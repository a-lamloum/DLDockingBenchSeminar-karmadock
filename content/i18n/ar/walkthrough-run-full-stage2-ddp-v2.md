# `scripts/run_full_stage2_ddp_v2.sh` {#scriptsrunfullstage2ddpv2sh}

## الغرض {#purpose}

هذه نسخة من مشغّل Stage-2 الأصلي بتقنية التدريب المتوازي الموزّع (DDP)، مع تغييرين صريحين في المعاملات الفائقة لمطابقة الورقة: لا اضمحلال للأوزان وpatience بقيمة 70. يحفظ الملف المنفصل النسخة التاريخية الأصلية ويجعل الفرق التجريبي قابلًا للتدقيق.

## موقعه في المنظومة {#how-it-fits-in}

يستخدم `condor/full_stage2_2gpu.sub` هذه النسخة ويمرر `S2_BATCH=8` و`S2_ACCUM=4` ونقطة تحقق Stage-1 النهائية، منتجًا دفعة عالمية فعّالة مقدارها 64 على وحدتي GPU. أما المسارات ومتغيرات البيئة والمدخلات وموقع الخرج وسلوك W&B والاختيار بين `torchrun` وعملية واحدة فتماثل `run_full_stage2_ddp.sh`.

## شرح الشيفرة {#walkthrough}

أُعيدت كتل الإعداد والمسارات لأنها شيفرة تنفيذية، مع أنها تطابق الأصل:

```bash
#!/usr/bin/env bash
# run_full_stage2_ddp_v2.sh — COPY ...
set -euo pipefail
set -x

KD=/app/KarmaDock
export PYTHONPATH=/app/KarmaDock:${PYTHONPATH:-}
SUBMIT="$PWD"

DATA="${FULL_DATA_DIR:-$HOME/repro_test/full_data}"
GRAPHS="${FULL_GRAPHS_DIR:-$HOME/repro_test/work_full}"
S2WORK="${S2_WORK_DIR:-$HOME/stage2_mgpu_test/work}"
STAGE1_CKPT="${STAGE1_CKPT:-$HOME/stage2_mgpu_test/stage1_snapshot.pkl}"
BS="${S2_BATCH:-8}"; ACC="${S2_ACCUM:-2}"
EP="${S2_EPOCHS:-1000}"

mkdir -p "$S2WORK"
[ -f "$STAGE1_CKPT" ] || { echo "ERROR: Stage-1 checkpoint not found: $STAGE1_CKPT"; exit 3; }

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NGPU:-0}" -lt 1 ] && NGPU=1
RUN="full_stage2_ddp_$(date +%Y%m%d-%H%M%S)"
echo "=== Stage-2 DDP on $NGPU GPU(s); effective batch = $((NGPU*BS*ACC)); run=$RUN ==="
```

يخدم النمط الصارم والمدخلات المنزلية المركّبة للقراءة فقط والخرج المعزول والتحقق من نقطة التحقق وعدّ GPU وتسمية التشغيل الأغراض نفسها الموضحة للأصل. ويصرّح الترويس أيضًا بخيار تصميم مهم: لا يُمرر `--jitter` إضافي لأن التحميل عند الطلب في مجموعة البيانات يعشوي وضعية بداية الربيطة بالفعل.

```bash
ARGS=( "$SUBMIT/train_ddp.py"
  --csv "$DATA/full_train.csv"   --graph_dir "$GRAPHS/full_train/graphs" --complex_dir "$GRAPHS/full_train/complex"
  --val_csv "$DATA/full_val.csv" --val_graph_dir "$GRAPHS/full_val/graphs"
  --out_dir "$S2WORK/ckpt_s2_ddp" --init_model "$STAGE1_CKPT" --pos_r 1
  --lr 1e-4 --weight_decay 0 --patience 70 --epochs "$EP"
  --batch_size "$BS" --accum_steps "$ACC" --num_workers 3 --random_seed 42 --resume
  --wandb --wandb_project karmadock-seminar --wandb_run_name "$RUN" )
```

هذا هو الفرق الوحيد المهم في سطر الأوامر: تزيل `--weight_decay 0` انكماش معاملات Adam الشبيه بـL2، وتسمح `--patience 70` بثبات أطول لخسارة التحقق. وتبقى بقية العناصر، ومنها هدف Stage-2 المشترك ومجموعة التحقق المنتقاة وسلوك الاستئناف، ثابتة لعزل أثر هذين الخيارين.

```bash
if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" "${ARGS[@]}"
else
  python3 -u "${ARGS[@]}"
fi
echo "=== Stage-2 DDP done: $S2WORK/ckpt_s2_ddp/karmadock_team002.pkl ==="
```

تفعّل عملية واحدة لكل GPU مرئية DDP، بينما تعود GPU واحدة إلى Python العادي. ومسار نقطة التحقق المطبوع في النهاية هو نموذج أفضل تحقق، لا نموذج الحقبة الأخيرة بالضرورة.

## محاذير وملاحظات {#gotchas--notes}

- تنطبق أيضًا كل محاذير التشغيل الأصلي: `nvidia-smi` تحت `pipefail`، وإعادة استخدام دليل الاستئناف، واعتماد الدفعة الفعّالة على world_size.
- «مطابق للورقة» هو المبرر المسجل في تعليقات البرنامج؛ ولا يتحقق المستودع برمجيًا من معاملات الورقة الفائقة.
- تكتب هذه النسخة والأصل إلى `S2WORK/ckpt_s2_ddp` الافتراضي نفسه، لذلك قد يجعل تشغيلهما بالقيم الافتراضية أحدهما يستأنف حالة الآخر.
