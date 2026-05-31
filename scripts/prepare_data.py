"""
prepare_data.py
Generates synthetic Arabic PII training data using a template-based approach.

"""
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

from entity_pools import (
    gen_person_name, gen_email, gen_phone, gen_iban,
    gen_account_number, gen_bank_account_number, gen_address,
    maybe_arabic_digits,
)


# TEMPLATES
# Each template is a string with {LABEL} placeholders.


TEMPLATES = [
    #  PERSON
    ("اسمي {PERSON} وأعمل في شركة كبيرة.", ["PERSON"]),
    ("أنا {PERSON} من القاهرة.", ["PERSON"]),
    ("سعدت بلقائك يا {PERSON}.", ["PERSON"]),
    ("الزميل {PERSON} سيحضر الاجتماع.", ["PERSON"]),
    ("My name is {PERSON} and I live here.", ["PERSON"]),
    ("هذا تقرير من إعداد {PERSON}.", ["PERSON"]),
    ("الدكتور {PERSON} متخصص في الجراحة.", ["PERSON"]),
    ("أرسل البريد إلى {PERSON} في الإدارة.", ["PERSON"]),
    ("المهندس {PERSON} يقود المشروع.", ["PERSON"]),
    ("تواصل مع {PERSON} للمزيد من التفاصيل.", ["PERSON"]),
    # Algeria dialect PERSON
    ("راني {PERSON} وهاذا رقمي.", ["PERSON"]),
    ("هذا هو الأستاذ {PERSON} من الجزائر.", ["PERSON"]),
    ("المتهم {PERSON} مثل أمام القضاء.", ["PERSON"]),
    ("وقّع العقد الأستاذ {PERSON} نيابةً عن الشركة.", ["PERSON"]),
    ("صرّح {PERSON} في تصريح صحفي.", ["PERSON"]),

    #  EMAIL
    ("بريدي الإلكتروني هو {EMAIL} للتواصل.", ["EMAIL"]),
    ("راسلني على {EMAIL} في أي وقت.", ["EMAIL"]),
    ("أرسلت الفاتورة إلى {EMAIL} أمس.", ["EMAIL"]),
    ("Please contact me at {EMAIL} for details.", ["EMAIL"]),
    ("البريد الرسمي للشركة هو {EMAIL}.", ["EMAIL"]),
    ("سجل دخولك باستخدام {EMAIL} وكلمة المرور.", ["EMAIL"]),
    ("للاستفسار: {EMAIL}", ["EMAIL"]),
    ("تم إرسال الرابط إلى {EMAIL}", ["EMAIL"]),
    ("Veuillez nous contacter à {EMAIL}.", ["EMAIL"]),
    ("أرسل طلبك عبر {EMAIL} قبل نهاية الأسبوع.", ["EMAIL"]),

    #  PHONE
    ("رقم هاتفي هو {PHONE_NUMBER} يمكنك الاتصال متى شئت.", ["PHONE_NUMBER"]),
    ("اتصل بي على {PHONE_NUMBER} مساءً.", ["PHONE_NUMBER"]),
    ("للحجز يرجى الاتصال بـ {PHONE_NUMBER}.", ["PHONE_NUMBER"]),
    ("My mobile is {PHONE_NUMBER}, call anytime.", ["PHONE_NUMBER"]),
    ("رقم الطوارئ: {PHONE_NUMBER}", ["PHONE_NUMBER"]),
    ("الواتساب الخاص بي {PHONE_NUMBER}.", ["PHONE_NUMBER"]),
    ("للتواصل اتصل على {PHONE_NUMBER} خلال ساعات العمل.", ["PHONE_NUMBER"]),
    ("سيتم إرسال رمز التحقق إلى {PHONE_NUMBER}.", ["PHONE_NUMBER"]),
    # Algeria dialect PHONE
    ("حوّل إليّ على الرقم {PHONE_NUMBER}.", ["PHONE_NUMBER"]),
    ("كلّمني على {PHONE_NUMBER} كي نتفاهم.", ["PHONE_NUMBER"]),
    ("رقم الاتصال الخاص بالوكالة هو {PHONE_NUMBER}.", ["PHONE_NUMBER"]),

    #  ADDRESS
    ("عنواني هو {ADDRESS} للتسليم.", ["ADDRESS"]),
    ("يقع المكتب في {ADDRESS}.", ["ADDRESS"]),
    ("توجه إلى {ADDRESS} غداً صباحاً.", ["ADDRESS"]),
    ("العنوان: {ADDRESS}", ["ADDRESS"]),
    ("الفرع الرئيسي في {ADDRESS}.", ["ADDRESS"]),
    ("التوصيل إلى {ADDRESS} مجاناً.", ["ADDRESS"]),
    # Algeria dialect ADDRESS
    ("العنوان: {ADDRESS}، الجزائر.", ["ADDRESS"]),
    ("الوكالة موجودة في {ADDRESS}.", ["ADDRESS"]),
    ("نسكن في {ADDRESS} منذ سنوات.", ["ADDRESS"]),
    ("يرجى التوجه إلى مقرنا في {ADDRESS}.", ["ADDRESS"]),

    #  IBAN
    ("رقم الآيبان هو {IBAN} للتحويل.", ["IBAN"]),
    ("حوّل المبلغ إلى الآيبان {IBAN}.", ["IBAN"]),
    ("IBAN: {IBAN}", ["IBAN"]),
    ("الآيبان الخاص بحسابي {IBAN}.", ["IBAN"]),
    ("استخدم {IBAN} لإتمام التحويل البنكي.", ["IBAN"]),
    ("Please send payment to IBAN {IBAN}.", ["IBAN"]),
    ("رقم الـ IBAN للتحويل الدولي: {IBAN}.", ["IBAN"]),

    #  ACCOUNT_NUMBER
    ("رقم الحساب هو {ACCOUNT_NUMBER} في النظام.", ["ACCOUNT_NUMBER"]),
    ("سجل برقم الحساب {ACCOUNT_NUMBER}.", ["ACCOUNT_NUMBER"]),
    ("الحساب رقم {ACCOUNT_NUMBER} نشط حالياً.", ["ACCOUNT_NUMBER"]),
    ("Your account ID is {ACCOUNT_NUMBER}.", ["ACCOUNT_NUMBER"]),
    ("معرف الحساب: {ACCOUNT_NUMBER}", ["ACCOUNT_NUMBER"]),
    ("رقم المشترك في المنظومة: {ACCOUNT_NUMBER}.", ["ACCOUNT_NUMBER"]),

    #  BANK_ACCOUNT_NUMBER
    ("رقم حسابي البنكي هو {BANK_ACCOUNT_NUMBER} في البنك الأهلي.", ["BANK_ACCOUNT_NUMBER"]),
    ("الحساب البنكي {BANK_ACCOUNT_NUMBER} يخص الشركة.", ["BANK_ACCOUNT_NUMBER"]),
    ("حوّل إلى الحساب البنكي رقم {BANK_ACCOUNT_NUMBER}.", ["BANK_ACCOUNT_NUMBER"]),
    ("Bank account: {BANK_ACCOUNT_NUMBER}", ["BANK_ACCOUNT_NUMBER"]),
    ("رقم الحساب المصرفي {BANK_ACCOUNT_NUMBER}.", ["BANK_ACCOUNT_NUMBER"]),
    ("رقم الحساب لدى بنك الجزائر الخارجي هو {BANK_ACCOUNT_NUMBER}.", ["BANK_ACCOUNT_NUMBER"]),
    ("يرجى الإيداع في الحساب رقم {BANK_ACCOUNT_NUMBER}.", ["BANK_ACCOUNT_NUMBER"]),

    # Extra PERSON
    ("تم تعيين {PERSON} مديراً للفرع الجديد.", ["PERSON"]),
    ("يسعدنا الإعلان عن انضمام {PERSON} إلى فريق العمل.", ["PERSON"]),
    ("أفاد {PERSON} بأن الوضع تحت السيطرة.", ["PERSON"]),
    ("قدّم {PERSON} استقالته من منصبه أمس.", ["PERSON"]),
    ("حضر {PERSON} الاجتماع ممثلاً عن الشركة.", ["PERSON"]),
    ("Please forward this to {PERSON} at your earliest convenience.", ["PERSON"]),
    ("The report was submitted by {PERSON}.", ["PERSON"]),

    # Extra EMAIL
    ("تلقّينا رسالة من {EMAIL} تطلب معلومات إضافية.", ["EMAIL"]),
    ("يمكنك تغيير كلمة المرور من خلال {EMAIL} المسجّل.", ["EMAIL"]),
    ("Send your CV to {EMAIL} before the deadline.", ["EMAIL"]),
    ("أُرسل تأكيد الحجز إلى {EMAIL} تلقائياً.", ["EMAIL"]),

    # Extra PHONE
    ("اتصلت بـ {PHONE_NUMBER} ولم يردّ أحد.", ["PHONE_NUMBER"]),
    ("رقم خدمة العملاء هو {PHONE_NUMBER} متاح ٢٤/٧.", ["PHONE_NUMBER"]),
    ("Please call {PHONE_NUMBER} to confirm your appointment.", ["PHONE_NUMBER"]),
    ("أضف {PHONE_NUMBER} إلى جهات اتصالك.", ["PHONE_NUMBER"]),
    ("تمّ إرسال كود التفعيل على {PHONE_NUMBER}.", ["PHONE_NUMBER"]),

    # Extra ADDRESS
    ("المقر الرئيسي للشركة في {ADDRESS}.", ["ADDRESS"]),
    ("يمكنك زيارتنا في {ADDRESS} من الأحد إلى الخميس.", ["ADDRESS"]),
    ("تمّ تسليم الطرد في {ADDRESS} بنجاح.", ["ADDRESS"]),
    ("Our headquarters are located at {ADDRESS}.", ["ADDRESS"]),
    ("سيتم افتتاح الفرع الجديد في {ADDRESS} الشهر القادم.", ["ADDRESS"]),

    # Extra IBAN
    ("يرجى التحقق من صحة الآيبان {IBAN} قبل التحويل.", ["IBAN"]),
    ("تمّ استلام المبلغ في الحساب {IBAN} بنجاح.", ["IBAN"]),
    ("أرسل لي الآيبان {IBAN} لإتمام عملية الدفع.", ["IBAN"]),
    ("Wire transfer to: {IBAN}", ["IBAN"]),

    # Extra ACCOUNT_NUMBER
    ("تمّ تجميد الحساب {ACCOUNT_NUMBER} لأسباب أمنية.", ["ACCOUNT_NUMBER"]),
    ("أدخل {ACCOUNT_NUMBER} للوصول إلى خدماتك.", ["ACCOUNT_NUMBER"]),
    ("Your reference number is {ACCOUNT_NUMBER}.", ["ACCOUNT_NUMBER"]),
    ("رقم ملفك لدينا هو {ACCOUNT_NUMBER}.", ["ACCOUNT_NUMBER"]),

    # Extra BANK_ACCOUNT_NUMBER
    ("تمّ تحويل الراتب إلى الحساب {BANK_ACCOUNT_NUMBER}.", ["BANK_ACCOUNT_NUMBER"]),
    ("يرجى إرفاق كشف حساب الرقم {BANK_ACCOUNT_NUMBER}.", ["BANK_ACCOUNT_NUMBER"]),
    ("الحساب {BANK_ACCOUNT_NUMBER} مفعّل ومتاح للاستخدام.", ["BANK_ACCOUNT_NUMBER"]),
    ("Please deposit to account number {BANK_ACCOUNT_NUMBER}.", ["BANK_ACCOUNT_NUMBER"]),
    ("رقم الحساب الجاري: {BANK_ACCOUNT_NUMBER}.", ["BANK_ACCOUNT_NUMBER"]),
    ("تحقّق من رصيد حسابك {BANK_ACCOUNT_NUMBER} عبر التطبيق.", ["BANK_ACCOUNT_NUMBER"]),

    #  MULTI-ENTITY
    ("اسمي {PERSON} ورقم تليفوني {PHONE_NUMBER}.", ["PERSON", "PHONE_NUMBER"]),
    ("أنا {PERSON} وبريدي {EMAIL}.", ["PERSON", "EMAIL"]),
    ("{PERSON} يسكن في {ADDRESS}.", ["PERSON", "ADDRESS"]),
    ("للتواصل مع {PERSON} على {PHONE_NUMBER} أو {EMAIL}.",
     ["PERSON", "PHONE_NUMBER", "EMAIL"]),
    ("رقم الآيبان {IBAN} لحساب {PERSON}.", ["IBAN", "PERSON"]),
    ("حوّل من {ACCOUNT_NUMBER} إلى الآيبان {IBAN}.", ["ACCOUNT_NUMBER", "IBAN"]),
    ("{PERSON}، {ADDRESS}، هاتف {PHONE_NUMBER}.",
     ["PERSON", "ADDRESS", "PHONE_NUMBER"]),
    ("صاحب الحساب {PERSON} ورقم الحساب البنكي {BANK_ACCOUNT_NUMBER}.",
     ["PERSON", "BANK_ACCOUNT_NUMBER"]),
    ("اسم العميل: {PERSON} - البريد: {EMAIL} - الجوال: {PHONE_NUMBER}",
     ["PERSON", "EMAIL", "PHONE_NUMBER"]),
    ("{PERSON} from {ADDRESS}, phone {PHONE_NUMBER}.",
     ["PERSON", "ADDRESS", "PHONE_NUMBER"]),
    # Algeria dialect MULTI-ENTITY
    ("راني {PERSON} وهاذا رقمي {PHONE_NUMBER}.", ["PERSON", "PHONE_NUMBER"]),
    ("المواطن {PERSON} يقيم في {ADDRESS} ورقم هاتفه {PHONE_NUMBER}.",
     ["PERSON", "ADDRESS", "PHONE_NUMBER"]),
    ("تواصل مع {PERSON} عبر {EMAIL} أو على الرقم {PHONE_NUMBER}.",
     ["PERSON", "EMAIL", "PHONE_NUMBER"]),
    ("حوّل المبلغ إلى {PERSON} عبر الآيبان {IBAN}.", ["PERSON", "IBAN"]),
    ("المستفيد: {PERSON}، الحساب: {BANK_ACCOUNT_NUMBER}، العنوان: {ADDRESS}.",
     ["PERSON", "BANK_ACCOUNT_NUMBER", "ADDRESS"]),
    ("يُرجى مراسلة {PERSON} على {EMAIL} أو زيارة مكتبه في {ADDRESS}.",
     ["PERSON", "EMAIL", "ADDRESS"]),
    # Extra MULTI-ENTITY
    ("رقم حساب {PERSON} البنكي هو {BANK_ACCOUNT_NUMBER}.", ["PERSON", "BANK_ACCOUNT_NUMBER"]),
    ("أرسل إلى {EMAIL} أو اتصل بـ {PHONE_NUMBER}.", ["EMAIL", "PHONE_NUMBER"]),
    ("عنوان {PERSON}: {ADDRESS}.", ["PERSON", "ADDRESS"]),
    ("تفاصيل العميل — الاسم: {PERSON}، الهاتف: {PHONE_NUMBER}، البريد: {EMAIL}، العنوان: {ADDRESS}.",
     ["PERSON", "PHONE_NUMBER", "EMAIL", "ADDRESS"]),
    ("حوّل من حساب {ACCOUNT_NUMBER} إلى {BANK_ACCOUNT_NUMBER}.",
     ["ACCOUNT_NUMBER", "BANK_ACCOUNT_NUMBER"]),
    ("معرّف المستخدم {ACCOUNT_NUMBER} مرتبط بالحساب البنكي {BANK_ACCOUNT_NUMBER}.",
     ["ACCOUNT_NUMBER", "BANK_ACCOUNT_NUMBER"]),
    ("رقم ملفك {ACCOUNT_NUMBER} والحساب البنكي للدفع هو {BANK_ACCOUNT_NUMBER}.",
     ["ACCOUNT_NUMBER", "BANK_ACCOUNT_NUMBER"]),
    ("Your user ID is {ACCOUNT_NUMBER} and your bank account is {BANK_ACCOUNT_NUMBER}.",
     ["ACCOUNT_NUMBER", "BANK_ACCOUNT_NUMBER"]),
    ("{PERSON} — IBAN: {IBAN} — Tel: {PHONE_NUMBER}.", ["PERSON", "IBAN", "PHONE_NUMBER"]),
    ("بيانات التحويل: المستفيد {PERSON}، الآيبان {IBAN}، البنك في {ADDRESS}.",
     ["PERSON", "IBAN", "ADDRESS"]),
    ("Contact {PERSON} at {EMAIL} or {PHONE_NUMBER}.", ["PERSON", "EMAIL", "PHONE_NUMBER"]),
    ("الحساب {BANK_ACCOUNT_NUMBER} باسم {PERSON} في {ADDRESS}.",
     ["BANK_ACCOUNT_NUMBER", "PERSON", "ADDRESS"]),
    ("للتسجيل أرسل بريدك {EMAIL} ورقمك {PHONE_NUMBER}.", ["EMAIL", "PHONE_NUMBER"]),
]


# HELD OUT VALIDATION TEMPLATES  never used in training
# Same difficulty mix as TEST: simple + complex + ambiguous + multi-entity
VAL_ONLY_TEMPLATES = [
    # PERSON — simple
    ("تمّت الموافقة على طلب {PERSON} من قِبَل الإدارة.", ["PERSON"]),
    ("The contract was signed by {PERSON} on behalf of the company.", ["PERSON"]),
    # PERSON — ambiguous context (title before name)
    ("نوّه التقرير إلى دور الأستاذ {PERSON} في إنجاح المشروع.", ["PERSON"]),
    # EMAIL — simple
    ("وصلنا بلاغ عبر {EMAIL} يفيد بوجود مشكلة تقنية.", ["EMAIL"]),
    # EMAIL — embedded mid-sentence
    ("يُرجى تأكيد بريدك الإلكتروني {EMAIL} لاستكمال التسجيل.", ["EMAIL"]),
    # PHONE — simple
    ("اترك رسالة على {PHONE_NUMBER} وسنعاود الاتصال بك.", ["PHONE_NUMBER"]),
    # PHONE — ambiguous (could be code)
    ("رمز التحقق الخاص بك سيُرسل إلى {PHONE_NUMBER}.", ["PHONE_NUMBER"]),
    # ADDRESS — simple
    ("تقع المحكمة الابتدائية في {ADDRESS}.", ["ADDRESS"]),
    # ADDRESS — complex embedded
    ("يُرجى إرسال المستندات بالبريد إلى {ADDRESS} قبل نهاية الأسبوع.", ["ADDRESS"]),
    # IBAN — simple
    ("يتم سداد الأقساط إلى الحساب الدولي {IBAN}.", ["IBAN"]),
    # IBAN — minimal context (hardest)
    ("رقم الآيبان: {IBAN}.", ["IBAN"]),
    # ACCOUNT_NUMBER — short ambiguous number
    ("سجّل شكواك باستخدام رقم ملفك {ACCOUNT_NUMBER}.", ["ACCOUNT_NUMBER"]),
    # ACCOUNT_NUMBER — alphanumeric
    ("تمّ ربط الجهاز برقم المشترك {ACCOUNT_NUMBER}.", ["ACCOUNT_NUMBER"]),
    # BANK_ACCOUNT_NUMBER — simple
    ("يتمّ صرف التعويض مباشرة في الحساب {BANK_ACCOUNT_NUMBER}.", ["BANK_ACCOUNT_NUMBER"]),
    # BANK_ACCOUNT_NUMBER — ambiguous (looks like ACCOUNT_NUMBER context)
    ("وردت حركة مشبوهة على الحساب رقم {BANK_ACCOUNT_NUMBER}.", ["BANK_ACCOUNT_NUMBER"]),
    # MULTI-ENTITY — 2 entities
    ("يُرجى تحويل المبلغ إلى {PERSON} على الحساب {BANK_ACCOUNT_NUMBER}.",
     ["PERSON", "BANK_ACCOUNT_NUMBER"]),
    ("أُحيل الملف إلى {PERSON} على {EMAIL} للمراجعة.", ["PERSON", "EMAIL"]),
    # MULTI-ENTITY — ACCOUNT vs BANK_ACCOUNT (hardest confusion pair)
    ("رقم التتبع {ACCOUNT_NUMBER} مرتبط بحساب {BANK_ACCOUNT_NUMBER}.",
     ["ACCOUNT_NUMBER", "BANK_ACCOUNT_NUMBER"]),
    # MULTI-ENTITY — 3 entities
    ("سيتواصل معك {PERSON} على {PHONE_NUMBER} أو {EMAIL}.",
     ["PERSON", "PHONE_NUMBER", "EMAIL"]),
    # MULTI-ENTITY — mixed Arabic/English
    ("{PERSON} lives at {ADDRESS}, phone: {PHONE_NUMBER}.",
     ["PERSON", "ADDRESS", "PHONE_NUMBER"]),
    # MULTI-ENTITY — 4 entities (hardest)
    ("بيانات العميل: {PERSON}، الهاتف {PHONE_NUMBER}، البريد {EMAIL}، العنوان {ADDRESS}.",
     ["PERSON", "PHONE_NUMBER", "EMAIL", "ADDRESS"]),
]


# HELD-OUT TEST TEMPLATES , never used in training or validation
# Same difficulty mix as VAL: simple + complex + ambiguous + multi-entity
TEST_ONLY_TEMPLATES = [
    # PERSON — simple
    ("يشرفني تقديم {PERSON} كمتحدث رئيسي في الملتقى.", ["PERSON"]),
    ("وردنا طلب من طرف {PERSON} بخصوص الملف.", ["PERSON"]),
    # PERSON — ambiguous context
    ("أكد مصدر مسؤول أن المدير {PERSON} سيتولى المنصب الجديد.", ["PERSON"]),
    # EMAIL — simple
    ("يمكن إرسال وثائق التسجيل إلى {EMAIL} مباشرة.", ["EMAIL"]),
    # EMAIL — embedded mid-sentence
    ("للتواصل مع الدعم الفني راسلنا على {EMAIL} في أي وقت.", ["EMAIL"]),
    # PHONE — simple
    ("يرجى إرسال رسالة نصية إلى {PHONE_NUMBER} لتأكيد الحضور.", ["PHONE_NUMBER"]),
    # PHONE — ambiguous (emergency number context)
    ("في حال الاستعجال يمكن الاتصال على {PHONE_NUMBER}.", ["PHONE_NUMBER"]),
    # ADDRESS — simple
    ("تجدنا في {ADDRESS} طوال أيام الأسبوع.", ["ADDRESS"]),
    # ADDRESS — complex embedded
    ("سيتم انعقاد الفعالية في {ADDRESS} على الساعة العاشرة صباحاً.", ["ADDRESS"]),
    # IBAN — simple
    ("يُرجى تحويل المبلغ المستحق إلى الرقم الدولي {IBAN}.", ["IBAN"]),
    # IBAN — minimal context (hardest)
    ("تفاصيل الحساب للتحويل: {IBAN}", ["IBAN"]),
    # ACCOUNT_NUMBER — short ambiguous number
    ("أدخل رقم حسابك {ACCOUNT_NUMBER} للمتابعة.", ["ACCOUNT_NUMBER"]),
    # ACCOUNT_NUMBER — alphanumeric
    ("تم تجميد الحساب {ACCOUNT_NUMBER} بسبب نشاط مريب.", ["ACCOUNT_NUMBER"]),
    # BANK_ACCOUNT_NUMBER — simple
    ("يُودَع الراتب شهرياً في الحساب رقم {BANK_ACCOUNT_NUMBER}.", ["BANK_ACCOUNT_NUMBER"]),
    # BANK_ACCOUNT_NUMBER — ambiguous context
    ("أرسل كشف حسابك للرقم {BANK_ACCOUNT_NUMBER} للتحقق.", ["BANK_ACCOUNT_NUMBER"]),
    # MULTI-ENTITY — 2 entities
    ("عزيزي {PERSON}، تم تحويل المبلغ إلى حسابك {BANK_ACCOUNT_NUMBER}.",
     ["PERSON", "BANK_ACCOUNT_NUMBER"]),
    ("للتسجيل أرسل بريدك {EMAIL} ورقمك {PHONE_NUMBER} إلى الإدارة.",
     ["EMAIL", "PHONE_NUMBER"]),
    # MULTI-ENTITY — ACCOUNT vs BANK_ACCOUNT (hardest confusion pair)
    ("معرّف المستخدم {ACCOUNT_NUMBER} مختلف عن رقم الحساب البنكي {BANK_ACCOUNT_NUMBER}.",
     ["ACCOUNT_NUMBER", "BANK_ACCOUNT_NUMBER"]),
    # MULTI-ENTITY — 3 entities
    ("المستفيد {PERSON} — آيبان: {IBAN} — هاتف: {PHONE_NUMBER}.",
     ["PERSON", "IBAN", "PHONE_NUMBER"]),
    # MULTI-ENTITY — mixed Arabic/English
    ("{PERSON} مقيم في {ADDRESS}، للتواصل: {EMAIL}.",
     ["PERSON", "ADDRESS", "EMAIL"]),
    # MULTI-ENTITY — 4 entities (hardest)
    ("اسم المستفيد: {PERSON}، الآيبان: {IBAN}، الهاتف: {PHONE_NUMBER}، العنوان: {ADDRESS}.",
     ["PERSON", "IBAN", "PHONE_NUMBER", "ADDRESS"]),
]


# NEGATIVE / DISTRACTOR TEMPLATES 
# Reduce false positives by exposing the model to non-PII numbers and entities
NEGATIVE_TEMPLATES = [
    "السعر هو 150 ريال فقط.",
    "تخفيضات تصل إلى 50% هذا الأسبوع.",
    "الاجتماع غداً الساعة 10 صباحاً.",
    "تأسست الشركة عام 1998.",
    "المنتج كود ABC123 متوفر في المخزن.",
    "درجة الحرارة اليوم 35 درجة.",
    "The meeting is at 3 PM tomorrow.",
    "حصلت على 92 درجة في الاختبار.",
    "الطلب رقم 45678 جاهز للاستلام.",
    "نسبة النمو وصلت إلى 12% هذا الربع.",
    "Code ABC-123 is for the new product line.",
    "العنوان الإلكتروني للموقع www.example.com.",
    "الفيلم مدته 120 دقيقة.",
    "تم بيع 5000 وحدة الشهر الماضي.",
    "رقم الفاتورة 99887 (وليس رقم حساب).",
    "في عام ٢٠٢٣ تخرجت من الجامعة.",
    # More distractors to cut false positives
    "المرسوم التنفيذي رقم 21-15 المؤرخ في 10 يناير 2021.",
    "الرمز البريدي للمنطقة هو 16000.",
    "الجلسة رقم 7 من الدورة الربعية.",
    "الفصل الثالث، المادة 45 من القانون الأساسي.",
    "سجل التجارة رقم 98/B/01234 صادر عن المركز الوطني.",
    "NIF: 000216100210345 — رقم تعريف جبائي (وليس حساباً).",
    "الرقم الجمركي للبضاعة: 4202.91.00.",
    "القانون رقم 90-11 المتعلق بعلاقات العمل.",
    "المرجع: 2024/DRH/045.",
    "The invoice total is 3,200 DZD.",
    "Réf. commande: CMD-20240512-007.",
    "السيارة لوحتها 123-456-07.",
    "رقم الطرد البريدي EE123456789DZ.",
    "وصل العدد إلى 1,250,000 مشترك.",
    "المؤشر ارتفع بنسبة 3.7 نقطة.",
    "الرقم التسلسلي للجهاز: SN-78A2C991.",
    # Hard negatives  look like PII but are not
    "الموقع الرسمي: https://www.interieur.gov.dz/index.php.",
    "رقم الضمان الاجتماعي للمؤسسة: 0799123456789.",
    "الرقم الجبائي: 00321600012345678 — يُستخدم للتصريح الضريبي.",
    "IP الخادم هو 192.168.1.254 في الشبكة الداخلية.",
    "رمز التحقق المرسل: 847291 — صالح لمدة 5 دقائق.",
    "الرقم المرجعي للملف: ALG-2024-00871.",
    "كود التحويل SWIFT: BNAADZZZXXX.",
    "رقم بطاقة الناخب: 160231450072.",
    "معرّف الجهاز: IMEI 354651089654321.",
    "رقم الوثيقة: D-2025/00143/MJ.",
    "الرقم التعريفي الوطني للمؤسسة (NIF): 099916001234567.",
    "رقم التسجيل في السجل التجاري: 16/00-1234567B19.",
    "تأكيد الحجز رقم BK-20240815-9923.",
    "رقم الرحلة: AH 1042 — الجزائر إلى باريس.",
    "الرمز الجمركي HS: 8471.30.00.00.",
    "معرف المعاملة: TXN-9982774410.",
    "رقم العقد: CTR/DZ/2024/0089.",
    "كود المنتج: DZ-MED-4421-B.",
    "رقم القضية أمام المحكمة: 2024/1423/جنح.",
    "الرقم الإحصائي للمنتج: 3004.90.99.00.",
]


# GENERATOR DISPATCH

GENERATORS = {
    "PERSON": gen_person_name,
    "EMAIL": gen_email,
    "PHONE_NUMBER": gen_phone,
    "ADDRESS": gen_address,
    "ACCOUNT_NUMBER": gen_account_number,
    "BANK_ACCOUNT_NUMBER": gen_bank_account_number,
    "IBAN": gen_iban,
}


def fill_template(template: str, labels: List[str]) -> Tuple[str, List[Dict]]:
    """
    Fill template placeholders with generated entities, tracking exact offsets
    in the FINAL string. Handles multiple placeholders correctly.
    """
    text_parts = []
    entities = []
    cursor = 0
    pos = 0

    while pos < len(template):
        # Find next placeholder
        next_brace = template.find("{", pos)
        if next_brace == -1:
            text_parts.append(template[pos:])
            break

        # Append literal text before the placeholder
        text_parts.append(template[pos:next_brace])
        cursor += (next_brace - pos)

        # Parse placeholder
        end_brace = template.find("}", next_brace)
        if end_brace == -1:
            raise ValueError(f"Unmatched brace in template: {template}")
        label = template[next_brace + 1:end_brace]

        if label not in GENERATORS:
            raise ValueError(f"Unknown label '{label}' in template: {template}")

        value = GENERATORS[label]()
        start = cursor
        end = cursor + len(value)
        entities.append({
            "start": start,
            "end": end,
            "label": label,
            "text": value,
        })
        text_parts.append(value)
        cursor = end
        pos = end_brace + 1

    final_text = "".join(text_parts)

    # Verification: every entity span must match the actual text
    for ent in entities:
        actual = final_text[ent["start"]:ent["end"]]
        assert actual == ent["text"], (
            f"Offset mismatch! Expected '{ent['text']}', got '{actual}' "
            f"in text: {final_text!r}"
        )
    return final_text, entities


def generate_example(template_pool: List[Tuple[str, List[str]]]) -> Dict:
    template, labels = random.choice(template_pool)
    text, entities = fill_template(template, labels)
    return {"text": text, "entities": entities}


def generate_negative_example() -> Dict:
    text = random.choice(NEGATIVE_TEMPLATES)
    return {"text": text, "entities": []}


def split_templates(templates, val_frac=0.1, test_frac=0.1, seed=42):
    """Split templates so test/val sets have UNSEEN templates but all entity types covered."""
    rng = random.Random(seed)

    # Group by entity types present
    from collections import defaultdict
    by_labels = defaultdict(list)
    for tpl in templates:
        key = tuple(sorted(set(tpl[1])))
        by_labels[key].append(tpl)

    train_tpl, val_tpl, test_tpl = [], [], []
    for key, group in by_labels.items():
        shuffled = group.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_test = max(1, int(n * test_frac))
        n_val = max(1, int(n * val_frac))
        test_tpl += shuffled[:n_test]
        val_tpl += shuffled[n_test:n_test + n_val]
        train_tpl += shuffled[n_test + n_val:]

    return train_tpl, val_tpl, test_tpl


def build_dataset(
    n_train: int = 25000,
    n_val: int = 2000,
    n_test: int = 2000,
    negative_ratio: float = 0.25,
    seed: int = 42,
):
    random.seed(seed)
    train_tpl, _, _ = split_templates(TEMPLATES, seed=seed)
    val_tpl = VAL_ONLY_TEMPLATES   
    test_tpl = TEST_ONLY_TEMPLATES 

    print(f"Templates -- train: {len(train_tpl)}, val: {len(val_tpl)} (held-out), test: {len(test_tpl)} (held-out)")

    def make_split(n: int, pos_tpl, name: str):
        n_neg = int(n * negative_ratio)
        n_pos = n - n_neg
        examples = []
        for _ in range(n_pos):
            examples.append(generate_example(pos_tpl))
        for _ in range(n_neg):
            examples.append(generate_negative_example())
        random.shuffle(examples)
        return examples

    train = make_split(n_train, train_tpl, "train")
    val = make_split(n_val, val_tpl, "val")
    test = make_split(n_test, test_tpl, "test")
    return train, val, test


def write_jsonl(examples: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  -> wrote {len(examples)} examples to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=25000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    train, val, test = build_dataset(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, seed=args.seed
    )
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "validation.jsonl")
    write_jsonl(test, out / "test.jsonl")

    # Quick stats
    from collections import Counter
    cnt = Counter()
    for ex in train:
        for e in ex["entities"]:
            cnt[e["label"]] += 1
    print("\nTrain entity counts:")
    for lbl, c in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"  {lbl}: {c}")


if __name__ == "__main__":
    main()
