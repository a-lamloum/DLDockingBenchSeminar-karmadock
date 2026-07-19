# 5. معلومات التدريب ومعاملاته (من الورقة) {#5-training-information--parameters-from-the-paper}

ينفّذ [`train.py`](scripts/train.py) *بروتوكول التدريب* في ورقة KarmaDock. حجم الدفعة الفعلي هو
64 في جميع التشغيلات (`batch_size 4 × accum_steps 16`)، وتقسيم التدريب والتحقق حتمي
(`val_frac 0.1`، و`seed 42`).

**P2 — التدريب من الصفر (2 stages وفق بروتوكول الورقة):**

| المرحلة | الهدف | `pos_r` | المحسّن | lr | weight_decay | patience |
|---|---|---|---|---|---|---|
| 1 | التسجيل / MDN فقط (تجاوز إرساء EGNN) | 0 | Adam | 1e-3 | 1e-5 | 70 |
| 2 | الإرساء + التسجيل (بدءًا من أفضل نتيجة للمرحلة 1) | 1 | Adam | 1e-4 | 1e-4 | 20 |

**P3 — الضبط الدقيق (إضافي، مرحلة واحدة، يبدأ من الأوزان المنشورة):**
`pos_r 1`، وAdam، و`lr 1e-4`، و`weight_decay 0`، و`patience 30`، وحجم دفعة فعلي 64،
و`val_frac 0.1`، و`seed 42`.

يمثل **pos_r** الوزن القياسي لخسارة RMSD، أي خسارة الإحداثيات والإرساء، في هدف تدريب
KarmaDock: ‏`training objective loss = pos_r * rmsd_loss + mdn_loss`. يعمل بوصفه مفتاحًا للتحسين الموضعي:
تكون قيمته 0 في المرحلة 1 لتدريب خسارة مسافات التفاعل MDN فقط، وتصبح 1 في المرحلة 2 لتفعيل حد
RMSD وتحسين إحداثيات الربيطة المتنبأ بها باتجاه الوضعية البلورية.

توجد منحنيات التدريب لكل حقبة في [`docs/p2_stage1_train_log.csv`](docs/p2_stage1_train_log.csv)،
و[`docs/p2_stage2_train_log.csv`](docs/p2_stage2_train_log.csv)، و
[`docs/p3_finetune_train_log.csv`](docs/p3_finetune_train_log.csv).

## ملخص سجلات التدريب {#training-log-summary}

حُسبت القيم مباشرةً من سجلات كل حقبة المرفقة، وتمثل القيم الأفضل الحدود الدنيا.

| التشغيل | الحقبات | أفضل خسارة تحقق | أفضل RMSD للتحقق | خسارة التحقق النهائية | الزمن المسجل | CSV |
|---|---:|---:|---:|---:|---:|---|
| Full-data Stage 2 | 470 | 3.55414 (epoch 399) | 2.81194 (epoch 399) | 3.62990 | 81.50 h | [`full_stage2_train_log.csv`](docs/full_stage2_train_log.csv) |
| P2 Stage 1 | 370 | 1.20531 (epoch 299) | N/A (`pos_r=0`) | 1.21701 | 1.95 h | [`p2_stage1_train_log.csv`](docs/p2_stage1_train_log.csv) |
| P2 Stage 2 | 146 | 6.16796 (epoch 125) | 4.94560 (epoch 125) | 6.24452 | 3.42 h | [`p2_stage2_train_log.csv`](docs/p2_stage2_train_log.csv) |
| P3 fine-tune | 68 | 3.48925 (epoch 37) | 2.58530 (epoch 37) | 3.59316 | 1.57 h | [`p3_finetune_train_log.csv`](docs/p3_finetune_train_log.csv) |
