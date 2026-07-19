# شروح الشيفرة {#code-walkthroughs}

هذه الملاحظات رفيق تعليمي للشيفرة التنفيذية في المستودع. تفترض معرفة بـPython وأساسيات تعلّم الآلة، لكنها تقدم مفاهيم الإرساء الجزيئي والتدريب الموزّع وHTCondor عندما تستخدمها الشيفرة. يقتبس كل شرح من المصدر ويوضح ما يفعله وكيف يعمل ولماذا وُجد.

## كيف تتكامل هذه الملفات؟ {#how-these-files-fit-together}

يسير التدفق الكامل كما يأتي:

1. **تحويل تخطيط المدخلات.** تطبّع `seminar_csv.py` مخططات CSV المدعومة. وتنسخ `convert_seminar_to_karmadock.py` كل زوج من الربيطة والبروتين إلى اصطلاح التسمية في KarmaDock الأصلي.
2. **المعالجة المسبقة والتدريب.** تستدعي أغلفة Shell البرنامجين الأصليين `pre_processing.py` و`generate_graph.py`، ثم `train.py` من هذا المستودع لعملية واحدة أو `train_ddp.py` لعملية واحدة لكل GPU. يمكن أن يبدأ التدريب عشوائيًا ببروتوكول التسجيل ثم الإرساء ذي المرحلتين، أو أن يضبط الأوزان المنشورة ضبطًا دقيقًا.
3. **الاستدلال.** يعالج `run_infer.sh` مجموعة اختبار مسبقًا ويستدعي `ligand_docking.py` الأصلي لإنتاج وضعيات خام ومخفّضة بحقل القوة ومحاذاة.
4. **إعادة تحويل التنبؤات.** ترتّب `convert_karmadock_to_seminar.py` التكرارات بحسب درجة MDN وتكتب ملفات `<id>_pred.sdf` بترتيب المقيّم.
5. **التقييم.** تحسب `evaluation.py` الجذر التربيعي لمتوسط مربع الانحراف (RMSD) للذرات الثقيلة بعد تصحيح التناظر، مع فحوص PoseBusters الاختيارية. وتطبّق `evaluate.sh` ذلك على جميع متغيرات مسارات النموذج الأولي ووضعياته.

تعمل ملفات `run_*.sh` كطبقات تنسيق قابلة لإعادة الإنتاج حول برامج Python هذه. وتقع ملفات `condor/` فوق أغلفة Shell؛ فهي تختار صورة Docker، وتجهّز المدخلات والمخرجات، وتحجز الموارد، وتمرر الوسائط ومتغيرات البيئة، وتوجّه السجلات.

## فهرس الشروح {#walkthrough-index}

| ملف المصدر | الشرح |
|---|---|
| `scripts/train.py` | [التدريب بعملية واحدة](scripts.train.md) |
| `scripts/train_ddp.py` | [التدريب الموزّع](scripts.train_ddp.md) |
| `scripts/seminar_csv.py` | [تطبيع مخططات CSV](scripts.seminar_csv.md) |
| `scripts/convert_seminar_to_karmadock.py` | [تحويل تخطيط المدخلات](scripts.convert_seminar_to_karmadock.md) |
| `scripts/convert_karmadock_to_seminar.py` | [تحويل تخطيط التنبؤات](scripts.convert_karmadock_to_seminar.md) |
| `evaluation/evaluation.py` | [تقييم RMSD وPoseBusters](evaluation.evaluation.md) |
| `scripts/run_train.sh` | [مشغّل تدريب النموذج الأولي](scripts.run_train.md) |
| `scripts/run_full_train.sh` | [مشغّل تدريب البيانات الكاملة](scripts.run_full_train.md) |
| `scripts/run_infer.sh` | [مشغّل الاستدلال](scripts.run_infer.md) |
| `scripts/run_full_stage2_ddp.sh` | [مشغّل Stage-2 الأصلي بتقنية DDP](scripts.run_full_stage2_ddp.md) |
| `scripts/run_full_stage2_ddp_v2.sh` | [مشغّل Stage-2 المطابق للورقة بتقنية DDP](scripts.run_full_stage2_ddp_v2.md) |
| `scripts/evaluate.sh` | [مشغّل التقييم الدفعي](scripts.evaluate.md) |
| جميع ملفات `condor/*.sub` الـ12 | [ملفات إرسال HTCondor](condor.md) |
