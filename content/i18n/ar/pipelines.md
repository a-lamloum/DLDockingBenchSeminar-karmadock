# 2. مسارات العمل الثلاثة {#2-the-three-pipelines}

| المسار | ما هو؟ | الأوزان |
|---|---|---|
| **P1 — خط الأساس** | استدلال فقط باستخدام `karmadock_screening.pkl` المنشور | من المشروع الأصلي داخل الصورة |
| **P2 — تدريب من الصفر** | بروتوكول الورقة ذي المرحلتين (2-stage)، مدرّب على `proto_train` (712) | [`model/p2_scratch_karmadock_team002.pkl`](model/p2_scratch_karmadock_team002.pkl) |
| **P3 — ضبط دقيق** *(إضافي)* | ضبط الأوزان المنشورة على `proto_train` (712) | [`model/p3_finetune_karmadock_team002.pkl`](model/p3_finetune_karmadock_team002.pkl) |

*الضبط الدقيق (P3) إضافة غير مطلوبة في جوهر المهمة، وما زال قيد التطوير.*

### سير العمل {#workflow}

**① التدريب — ينتج نقطتي تحقق P2 وP3:**

<a href="docs/workflow_training.png"><img src="docs/workflow_training.png" alt="سير عمل التدريب" width="580"></a>

*تُعالج `proto_train` (712) إلى رسوم بيانية، ثم يبدأ التدريب. يحدد `--init_model` المسار: يبدأ
**P2** من الصفر (المرحلة 1 لتسجيل MDN، ثم المرحلة 2 مع RMSD للإرساء)، بينما يضبط **P3** الأوزان
المنشورة. تحتفظ `Early_stopper` بأفضل حقبة بوصفها نقطة التحقق.*

**② الاستدلال والتقييم:**

<a href="docs/workflow_inference.png"><img src="docs/workflow_inference.png" alt="سير عمل الاستدلال والتقييم" width="700"></a>

*يُشغّل المسار مرة لكل من P1 وP2 وP3: تُعالج `proto_test` (136)، ثم تُرسى الربيطات وتُمنح درجات
بالأوزان المختارة، وتُصدّر متغيرات الوضعية 3 (الخام وFF والمحاذاة)، ثم تُقيّم بواسطة
`evaluation.py` الرسمي باستخدام RMSD المصحح للتناظر وأفضل وضعية top-1.*
