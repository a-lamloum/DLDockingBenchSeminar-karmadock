# شرح `scripts/run_full_stage2_ddp.sh` {#scriptsrunfullstage2ddpsh}

## الغرض {#purpose}

هذا مشغّل معزول للمرحلة 2 على البيانات الكاملة باستخدام التدريب المتوازي الموزّع (DDP) في PyTorch. يعيد استخدام أوزان المرحلة 1 للقراءة فقط ورسومًا بيانية مبنية مسبقًا، ويكتب في مساحة عمل منفصلة، مما يتيح تجارب متعددة GPU من دون لمس تشغيل التدريب الرئيس.

## موقعه في المنظومة {#how-it-fits-in}

ينقل `condor/full_stage2_mgpu.sub` و`condor/smoke_2gpu.sub` هذا البرنامج بجوار `train_ddp.py` و`seminar_csv.py`. تكون المدخلات افتراضيًا في مسارات ضمن مجلد المنزل المثبت، ويمكن تجاوزها بواسطة `FULL_DATA_DIR` و`FULL_GRAPHS_DIR` و`S2_WORK_DIR` و`STAGE1_CKPT` و`S2_BATCH` و`S2_ACCUM` و`S2_EPOCHS`. الخرج هو `$S2_WORK_DIR/ckpt_s2_ddp/` إضافة إلى سجلات W&B.

## شرح الشيفرة {#walkthrough}

```bash
#!/usr/bin/env bash
# ...
set -euo pipefail
set -x

KD=/app/KarmaDock
export PYTHONPATH=/app/KarmaDock:${PYTHONPATH:-}
SUBMIT="$PWD"
```

يساعد سلوك Bash الصارم وتسجيل التتبع على تشخيص إخفاقات العمليات المتعددة. تصبح نسخة KarmaDock الأصلية قابلة للاستيراد، ويُتوقع وجود الملفات المنقولة في جذر صندوق المهمة.

```bash
DATA="${FULL_DATA_DIR:-$HOME/repro_test/full_data}"
GRAPHS="${FULL_GRAPHS_DIR:-$HOME/repro_test/work_full}"
S2WORK="${S2_WORK_DIR:-$HOME/stage2_mgpu_test/work}"
STAGE1_CKPT="${STAGE1_CKPT:-$HOME/stage2_mgpu_test/stage1_snapshot.pkl}"
BS="${S2_BATCH:-8}"; ACC="${S2_ACCUM:-2}"
EP="${S2_EPOCHS:-1000}"

mkdir -p "$S2WORK"
[ -f "$STAGE1_CKPT" ] || { echo "ERROR: Stage-1 checkpoint not found: $STAGE1_CKPT"; exit 3; }
```

تُبقي القيم الافتراضية المدخلات الكبيرة في مجلد المنزل المثبت، وكل المخرجات في شجرة `stage2_mgpu_test` المعزولة. تطبع المجموعة `{ ...; }` خطأ واضحًا وتخرج بالحالة 3 إذا غابت نقطة تحقق التهيئة المجمدة.

```bash
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NGPU:-0}" -lt 1 ] && NGPU=1
RUN="full_stage2_ddp_$(date +%Y%m%d-%H%M%S)"
echo "=== Stage-2 DDP on $NGPU GPU(s); effective batch = $((NGPU*BS*ACC)); run=$RUN ==="
```

يطبع `nvidia-smi -L` سطرًا لكل GPU مرئي. يعرض التعبير الحسابي حجم الدفعة الفعلي العالمي بوصفه عدد العمليات × الدفعة الفيزيائية لكل عملية × خطوات التجميع. يجعل الطابع الزمني اسم W&B الأساسي المقروء فريدًا. لا ينشئ الرجوع إلى قيمة واحدة GPU؛ بل يختار فرع العملية الواحدة ويترك `train_ddp.py` يقرر بين CUDA وCPU.

```bash
ARGS=( "$SUBMIT/train_ddp.py"
  --csv "$DATA/full_train.csv"   --graph_dir "$GRAPHS/full_train/graphs" --complex_dir "$GRAPHS/full_train/complex"
  --val_csv "$DATA/full_val.csv" --val_graph_dir "$GRAPHS/full_val/graphs"
  --out_dir "$S2WORK/ckpt_s2_ddp" --init_model "$STAGE1_CKPT" --pos_r 1
  --lr 1e-4 --weight_decay 1e-4 --patience 20 --epochs "$EP"
  --batch_size "$BS" --accum_steps "$ACC" --num_workers 3 --random_seed 42 --resume
  --wandb --wandb_project karmadock-seminar --wandb_run_name "$RUN" )
```

تحافظ المصفوفة على حدود الوسائط بأمان. تفعّل `pos_r=1` التدريب المشترك للإرساء وRMSD وتسجيل MDN. تستخدم هذه النسخة الأصلية weight_decay بقيمة `1e-4` وpatience بقيمة 20. وتحافظ مدخلات التدريب والتحقق الصريحة على التقسيم المنقح.

```bash
if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nnodes=1 --nproc_per_node="$NGPU" "${ARGS[@]}"
else
  python3 -u "${ARGS[@]}"
fi
echo "=== Stage-2 DDP done: $S2WORK/ckpt_s2_ddp/karmadock_team002.pkl ==="
```

يبدأ `torchrun` عملية Python واحدة لكل GPU، ويمرر `RANK` و`LOCAL_RANK` و`WORLD_SIZE`؛ ويستخدمها `train_ddp.py` لتهيئة NCCL وDDP. ينشئ `--standalone --nnodes=1` نقطة الالتقاء محليًا على جهاز واحد. يتجنب تخصيص GPU واحد إعداد مجموعة العمليات ويستخدم مسار العملية الواحدة في البرنامج.

## تنبيهات وملاحظات {#gotchas--notes}

- يُستخدم `nvidia-smi` داخل برنامج يفعّل `pipefail`. إذا غاب الأمر أو أخفق بدل أن يعرض صفر أجهزة، فقد ينهي خط الإسناد البرنامج قبل تنفيذ الرجوع الاحتياطي.
- يعيد `--resume` استخدام `$S2WORK/ckpt_s2_ddp/last.pt`؛ لذلك قد يتابع استخدام مجلد الخرج نفسه تشغيلًا سابقًا رغم امتلاك `RUN` طابعًا زمنيًا جديدًا.
- تعطي القيم الافتراضية `BS=8` و`ACC=2` دفعة عالمية 64 فقط مع أربع وحدات GPU؛ ويلزم تجاوز ملف الإرسال لأعداد أخرى.
