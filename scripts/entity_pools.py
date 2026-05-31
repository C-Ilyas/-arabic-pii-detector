"""
Entity value pools for synthetic Arabic PII data generation.
Provides realistic values for each PII type with both Arabic and Western digits.
"""
import random
import string


# DIGIT CONVERSION
ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
WESTERN_DIGITS = "0123456789"
W2A = str.maketrans(WESTERN_DIGITS, ARABIC_INDIC_DIGITS)
A2W = str.maketrans(ARABIC_INDIC_DIGITS, WESTERN_DIGITS)


def to_arabic_digits(s: str) -> str:
    return s.translate(W2A)


def maybe_arabic_digits(s: str, p: float = 0.35) -> str:
    """Randomly convert digits to Arabic-Indic with probability p."""
    return to_arabic_digits(s) if random.random() < p else s


# PERSON NAMES
ARABIC_FIRST_NAMES = [
    "محمد", "أحمد", "علي", "حسن", "حسين", "إبراهيم", "يوسف", "عمر", "خالد",
    "عبدالله", "عبدالرحمن", "عبدالعزيز", "سعيد", "كريم", "طارق", "مصطفى",
    "محمود", "سامي", "فهد", "ناصر", "ياسر", "وليد", "هشام", "أيمن", "زياد",
    "بلال", "أنس", "زيد", "فيصل", "ماجد", "بدر", "سلطان", "نواف", "رامي",
    "فاطمة", "عائشة", "خديجة", "مريم", "زينب", "سارة", "نور", "هدى", "آية",
    "ريم", "لينا", "دانا", "سلمى", "ليلى", "هند", "أمل", "إيمان", "رنا",
    "نادية", "سميرة", "فدوى", "غادة", "هالة", "منى", "وفاء", "ياسمين",
    "عبدالقادر", "رضا", "لخضر", "مختار", "الطاهر", "نورالدين", "عمر", "رابح",
    "بوعلام", "الحسين", "إسماعيل", "وهيب", "جمال", "كمال", "سفيان",
    "نسرين", "سهيلة", "وردة", "حياة", "نجوى", "فيروز", "زهرة", "أسماء",
    "صونية", "كريمة", "لويزة", "خيرة", "نوال", "رحمة", "سعاد",
]
ARABIC_LAST_NAMES = [
    "العتيبي", "الشمري", "القحطاني", "الدوسري", "الزهراني", "الغامدي",
    "المالكي", "الحربي", "السبيعي", "البلوي", "الرشيدي", "السلمي",
    "حسن", "محمود", "إبراهيم", "علي", "عبدالله", "كمال", "نجيب", "رضا",
    "الحسيني", "الصباح", "النجار", "الحداد", "الخطيب", "الحمادي", "المنصور",
    "خوري", "حداد", "شاهين", "ناصيف", "أبوزيد", "البدوي", "الفاسي",
    "بن علي", "بن عمر", "بن محمد", "بوزيد", "بوعزيز", "بلقاسم", "بلحاج",
    "بن يوسف", "بن عبدالله", "حمداني", "زروالي", "مزياني", "قاسمي",
    "بوطالب", "بن شيخ", "حمودة", "بوشامة", "غريب", "سلطاني", "عمروش",
    "مداني", "بلعباس", "زيتوني", "شريف", "بوخاري", "حداد", "كربوسة",
]
TRANSLITERATED_FIRST = [
    "Mohamed", "Ahmed", "Ali", "Omar", "Khaled", "Youssef", "Karim",
    "Fatima", "Aisha", "Sara", "Layla", "Nour", "Yara", "Rana", "Lina",
    "Hassan", "Mahmoud", "Tarek", "Amr", "Bassem", "Mostafa", "Hesham",
    "Yazid", "Sofiane", "Nassim", "Rédha", "Yacine", "Bilal", "Amine",
    "Meriem", "Nawel", "Dalila", "Sonia", "Warda", "Loubna", "Asma",
]
TRANSLITERATED_LAST = [
    "Hassan", "Ahmed", "Mohamed", "Ali", "Mahmoud", "Ibrahim",
    "Mansour", "Khalil", "Saleh", "Farouk", "Nasser", "Rashid",
    "Benali", "Bouzid", "Hamidi", "Zerrouki", "Meziane", "Bouaziz",
    "Belkacem", "Hadj", "Saadi", "Cherif", "Boukerche",
]


def gen_person_name() -> str:
    style = random.random()
    if style < 0.65:
        # Arabic name, 2-3 components
        n_parts = random.choices([2, 3], weights=[0.7, 0.3])[0]
        parts = [random.choice(ARABIC_FIRST_NAMES)]
        for _ in range(n_parts - 1):
            parts.append(random.choice(ARABIC_FIRST_NAMES + ARABIC_LAST_NAMES))
        return " ".join(parts)
    elif style < 0.85:
        # Transliterated English name
        return f"{random.choice(TRANSLITERATED_FIRST)} {random.choice(TRANSLITERATED_LAST)}"
    else:
        # Mixed: Arabic first + transliterated last 
        return f"{random.choice(ARABIC_FIRST_NAMES)} {random.choice(TRANSLITERATED_LAST)}"


# EMAILS
EMAIL_DOMAINS = [
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "example.com", "test.com", "company.sa", "mail.eg", "uae.ae",
    "protonmail.com", "live.com", "icloud.com",
    "yahoo.fr", "hotmail.fr", "gmail.com", "mail.dz", "company.dz",
    "esi.dz", "univ-alger.dz",
]
EMAIL_USER_PARTS = [
    "mohamed", "ahmed", "ali", "sara", "fatima", "user", "info",
    "contact", "support", "admin", "hello", "test", "m.hassan",
    "a.ali", "y.ibrahim", "karim85", "sara_2020", "the.developer",
]


def gen_email() -> str:
    user = random.choice(EMAIL_USER_PARTS)
    if random.random() < 0.3:
        user = user + str(random.randint(1, 9999))
    if random.random() < 0.2:
        user = user.replace(".", "_")
    return f"{user}@{random.choice(EMAIL_DOMAINS)}"


# PHONE NUMBERS
def gen_phone() -> str:
    style = random.random()
    if style < 0.28:
        # Egyptian mobile: 010/011/012/015 + 8 digits
        prefix = random.choice(["010", "011", "012", "015"])
        phone = prefix + "".join(random.choices("0123456789", k=8))
    elif style < 0.43:
        # Saudi: +966 5XX XXX XXXX
        body = "5" + "".join(random.choices("0123456789", k=8))
        sep = random.choice(["", " ", "-"])
        phone = f"+966{sep}{body}"
    elif style < 0.56:
        # UAE: +971 5X XXX XXXX
        body = "5" + random.choice("01246789") + "".join(random.choices("0123456789", k=7))
        sep = random.choice(["", " ", "-"])
        phone = f"+971{sep}{body}"
    elif style < 0.72:
        # Algeria mobile: +213 5XX/6XX/7XX XXX XXX (or local 05/06/07)
        mobile_prefix = random.choice(["5", "6", "7"])
        body = mobile_prefix + "".join(random.choices("0123456789", k=8))
        sep = random.choice(["", " ", "-"])
        if random.random() < 0.5:
            phone = f"+213{sep}{body}"
        else:
            phone = f"0{body}"
    elif style < 0.85:
        # International generic
        phone = "+" + "".join(random.choices("0123456789", k=random.randint(10, 13)))
    else:
        # Landline style with dashes/spaces
        phone = f"02-{random.randint(2000,9999)}-{random.randint(1000,9999)}"
    return maybe_arabic_digits(phone, p=0.30)


# IBAN 
IBAN_LENGTHS = {
    "EG": 29,  # Egypt
    "SA": 24,  # Saudi Arabia
    "AE": 23,  # UAE
    "JO": 30,  # Jordan
    "KW": 30,  # Kuwait
    "QA": 29,  # Qatar
    "BH": 22,  # Bahrain
    "DZ": 26,  # Algeria
}


def _iban_mod97(s: str) -> int:
    """Compute mod 97 for IBAN checksum validation."""
    digits = ""
    for ch in s:
        if ch.isdigit():
            digits += ch
        else:
            digits += str(ord(ch.upper()) - 55)
    rem = 0
    for d in digits:
        rem = (rem * 10 + int(d)) % 97
    return rem


def gen_iban() -> str:
    country = random.choice(list(IBAN_LENGTHS.keys()))
    total_len = IBAN_LENGTHS[country]
    bban_len = total_len - 4  
    bban = "".join(random.choices("0123456789", k=bban_len))
    # Compute valid checksum
    rearranged = bban + country + "00"
    check = 98 - _iban_mod97(rearranged)
    iban = f"{country}{check:02d}{bban}"
    # Sometimes add spaces every 4 chars 
    if random.random() < 0.25:
        iban = " ".join(iban[i:i+4] for i in range(0, len(iban), 4))
    return iban


# ACCOUNT NUMBERS

def gen_account_number() -> str:
    """Short internal account number — typically 6-10 chars, may include letters."""
    style = random.random()
    if style < 0.5:
        # Pure digits, 6-10 long
        n = random.randint(6, 10)
        acc = "".join(random.choices("0123456789", k=n))
    elif style < 0.85:
        # Mixed alphanumeric, 6-9 chars
        n = random.randint(6, 9)
        acc = "".join(random.choices(string.ascii_uppercase + "0123456789", k=n))
    else:
        # Prefixed
        prefix = random.choice(["ACC", "A", "C"])
        acc = prefix + "".join(random.choices("0123456789", k=random.randint(5, 8)))
    return maybe_arabic_digits(acc, p=0.20)


def gen_bank_account_number() -> str:
    """Bank-issued account number — longer, usually 10-18 digits."""
    n = random.randint(10, 18)
    acc = "".join(random.choices("0123456789", k=n))
    # Sometimes formatted with dashes
    if random.random() < 0.2:
        acc = f"{acc[:4]}-{acc[4:8]}-{acc[8:]}"
    return maybe_arabic_digits(acc, p=0.25)


# ADDRESSES
STREETS = [
    "شارع التحرير", "شارع الجمهورية", "شارع الملك فهد", "شارع العروبة",
    "شارع النيل", "شارع الأهرام", "شارع الهرم", "شارع رمسيس",
    "شارع الملك فيصل", "شارع الأمير محمد بن عبدالعزيز", "شارع المعز",
    "شارع الكورنيش", "شارع الحرية", "شارع الاستقلال", "شارع الجلاء",
    "شارع أول نوفمبر", "شارع العقيد لطفي", "شارع زيغود يوسف",
    "شارع الشهداء", "شارع ديدوش مراد", "شارع بن مهيدي",
    "شارع العربي بن مهيدي", "شارع محمد بلوزداد", "شارع محمد بيوض",
]
DISTRICTS = [
    "الدقي", "المهندسين", "مدينة نصر", "المعادي", "الزمالك", "الحي العاشر",
    "العليا", "الملز", "النخيل", "الروضة", "السلام", "النزهة",
    "الجميرا", "ديرة", "البرشاء", "المرقبات", "النهضة",
    "باب الوادي", "القصبة", "المدنية", "بئر مراد رايس", "دالي إبراهيم",
    "بن عكنون", "الحراش", "باب الزوار", "برج البحري", "حيدرة",
    "سيدي امحمد", "الأبيار", "العاشور", "القبة", "المحمدية",
]
CITIES = [
    "القاهرة", "الجيزة", "الإسكندرية", "أسيوط", "المنصورة", "طنطا",
    "الرياض", "جدة", "الدمام", "مكة", "المدينة المنورة", "الطائف",
    "دبي", "أبوظبي", "الشارقة", "العين",
    "عمّان", "إربد", "الزرقاء", "الكويت", "الدوحة", "المنامة",
    "الجزائر العاصمة", "وهران", "قسنطينة", "عنابة", "بلعباس",
    "باتنة", "جيجل", "سطيف", "سكيكدة", "بجاية", "تلمسان",
    "تيزي وزو", "البليدة", "مستغانم", "المدية", "بسكرة",
    "الأغواط", "ورقلة", "تمنراست", "أدرار",
]


def gen_address() -> str:
    style = random.random()
    if style < 0.5:
        n = random.randint(1, 250)
        n_str = maybe_arabic_digits(str(n), p=0.4)
        addr = f"{n_str} {random.choice(STREETS)}، {random.choice(DISTRICTS)}، {random.choice(CITIES)}"
    elif style < 0.8:
        addr = f"{random.choice(STREETS)}، {random.choice(DISTRICTS)}، {random.choice(CITIES)}"
    else:
        building = random.randint(1, 99)
        flat = random.randint(1, 20)
        building_s = maybe_arabic_digits(str(building), p=0.3)
        flat_s = maybe_arabic_digits(str(flat), p=0.3)
        addr = f"عمارة {building_s} شقة {flat_s} {random.choice(STREETS)}، {random.choice(CITIES)}"
    return addr
