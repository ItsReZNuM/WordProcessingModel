# Persian Text Preprocessing Module
# Description:
#   An advanced, open-source-ready Persian text preprocessor
#   designed for all content types (news, scientific, chats, skills, motivational, etc.)
#   It handles emojis, English→Persian, Finglish→Persian, typos,
#   scientific tokens, kashida, numbers, laughter, links, and more.
#   Expandable dictionaries allow easy customization.


import re


class PersianPreprocessor:

    def __init__(self):

        # -------------------------------------------------
        # Emoji → Word Mapping 
        # -------------------------------------------------
        self.emoji_map = {
            "😂": "خنده",
            "🤣": "خنده شدید",
            "😅": "خنده عصبی",
            "😊": "لبخند",
            "🙂": "لبخند",
            "😉": "چشمک",
            "😁": "خنده",
            "😢": "غم",
            "😭": "گریه",
            "😡": "عصبانیت",
            "🤬": "خشم",
            "😠": "عصبانیت",
            "😒": "ناامیدی",
            "😐": "بی‌تفاوتی",
            "😍": "علاقه",
            "❤️": "عشق",
            "💔": "قلب_شکسته",
            "🔥": "هیجان",
            "👍": "تایید",
            "👎": "عدم_تایید",
            "🙏": "تشکر",
            "👏": "تشویق",
            "🤯": "شگفتی",
            "🤔": "تفکر",
            "😴": "خواب",
            "🤮": "حالت_بد",
            "🧠": "مغز",
            "🔬": "میکروسکوپ",
            "🧪": "آزمایش",
            "📈": "افزایش",
            "📉": "کاهش",
            "⚡": "الکتریسیته",
            "🌡": "دما",
            "☀️": "خورشید",
            "🌕": "ماه",
            "🌍": "زمین",
            "🧬": "دی‌ان‌ای",
        }

        # -------------------------------------------------
        # English → Persian Dictionary
        # -------------------------------------------------
        self.eng_to_fa = {
            "cool": "باحال",
            "nice": "خوب",
            "great": "عالی",
            "perfect": "بی‌نقص",
            "amazing": "شگفت‌انگیز",
            "price": "قیمت",
            "game": "بازی",
            "movie": "فیلم",
            "sorry": "ببخشید",
            "ok": "باشه",
            "thanks": "مرسی",
            "thankyou": "مرسی",
            "lol": "خنده",
            "wtf": "چی",
            "good": "خوب",
            "bad": "بد",
            "update": "آپدیت",
            "bug": "باگ",
            "error": "خطا",
            "delete": "حذف",
            "free": "رایگان",
            "faster": "سریع‌تر",
            "slow": "کند",
            "love": "عشق",
            "hate": "نفرت",
            "support": "پشتیبانی",
            "server": "سرور",
            "account": "اکانت",
            "login": "ورود",
            "logout": "خروج",
            "install": "نصب",
            "download": "دانلود",
            "dna": "دی‌ان‌ای",
            "co2": "دی‌اکسید_کربن",
            "uv": "فرابنفش",
            "ir": "مادون_قرمز",
        }

        # -------------------------------------------------
        # Finglish → Persian Dictionary
        # -------------------------------------------------
        self.finglish_to_fa = {
            "salam": "سلام",
            "chetori": "چطوری",
            "khubam": "خوبم",
            "khobam": "خوبم",
            "khoobam": "خوبم",
            "khoob": "خوب",
            "khob": "خوب",
            "man": "من",
            "kheyli": "خیلی",
            "ali": "عالی",
            "eshgh": "عشق",
            "bebakhshid": "ببخشید",
            "merci": "مرسی",
            "lotfan": "لطفاً",
            "mamnoon": "ممنون",
            "khafan": "خفن",
            "bahal": "باحال",
            "khoda": "خدا",
            "khastam": "خستم",
            "bad": "بد",
            "khob": "خوب",
            "fekr": "فکر",
        }


        # -------------------------------------------------
        # Persian shortcuts 
        # -------------------------------------------------
        self.shortcuts = {
            "خخ": "خنده",
            "خخخ": "خنده",
            "خخخخ": "خنده",
            "هه": "خنده",
            "ههه": "خنده",
            "هههه": "خنده",
            ":)": "خنده",
            ":))": "خنده",
            ":)))": "خنده",
            ":((": "غم",
            ":((": "غم",
            ":(": "غم",
        }

        # -------------------------------------------------
        # Typo corrections
        # -------------------------------------------------
        self.typos = {
            "میخوام": "می‌خوام",
            "میخام": "می‌خوام",
            "نمیخوام": "نمی‌خوام",
            "میخواهم": "می‌خواهم",
            "واقعا": "واقعاً",
            "بخاطر": "به‌خاطر",
            "کتاب خانه": "کتاب‌خانه",
        }

        # -------------------------------------------------
        # Scientific symbol replacements
        # -------------------------------------------------
        self.science_tokens = {
            "°C": "درجه_سانتی‌گراد",
            "°F": "درجه_فارنهایت",
            "km": "کیلومتر",
            "kg": "کیلوگرم",
            "mg": "میلی‌گرم",
            "H2O": "آب",
            "CO2": "دی‌اکسید_کربن",
            "DNA": "دی‌ان‌ای",
            "UV": "فرابنفش",
            "IR": "مادون_قرمز",
        }


    # -------------------------------
    # Replace links
    # -------------------------------
    def replace_links(self, text):
        return re.sub(r'https?://\S+|www\.\S+', ' لینک ', text)

    # -------------------------------
    # Emoji replacement
    # -------------------------------
    def replace_emojis(self, text):
        for e, w in self.emoji_map.items():
            text = text.replace(e, f" {w} ")
        return text

    # -------------------------------
    # English → Persian
    # -------------------------------
    def replace_english(self, text):
        words = text.split()
        new = []
        for w in words:
            key = w.lower().strip(".,!?:;")
            new.append(self.eng_to_fa.get(key, w))
        return " ".join(new)

    # -------------------------------
    # Finglish → Persian
    # -------------------------------
    def replace_finglish(self, text):
        words = text.split()
        new = []
        for w in words:
            lw = w.lower()
            new.append(self.finglish_to_fa.get(lw, w))
        return " ".join(new)

    # -------------------------------
    # Persian shortcuts (خخخ → خنده)
    # -------------------------------
    def replace_shortcuts(self, text):
        for k,v in self.shortcuts.items():
            text = text.replace(k, f" {v} ")
        return text

    # -------------------------------
    # Typos
    # -------------------------------
    def fix_typos(self, text):
        for k,v in self.typos.items():
            text = text.replace(k, v)
        return text

    # -------------------------------
    # Normalize laughs
    # -------------------------------
    def normalize_laughs(self, text):
        text = re.sub(r':\)+', ' خنده ', text)
        text = re.sub(r'(ه|خ){3,}', ' خنده ', text)
        return text

    # -------------------------------
    # English digits → Persian
    # -------------------------------
    def convert_numbers(self, text):
        eng = "0123456789"
        fa  = "۰۱۲۳۴۵۶۷۸۹"
        return text.translate(str.maketrans(eng, fa))

    # -------------------------------
    # Scientific Symbols
    # -------------------------------
    def replace_science(self, text):
        for k,v in self.science_tokens.items():
            text = text.replace(k, f" {v} ")
        return text

    # -------------------------------
    # Remove keshide
    # -------------------------------
    def remove_keshide(self, text):
        return re.sub(r'(.)\1{2,}', r'\1', text)

    # -------------------------------
    # Fix Arabic chars
    # -------------------------------
    def fix_arabic(self, text):
        return text.replace("ي", "ی").replace("ك", "ک")

    # -------------------------------
    # MASTER PIPELINE
    # -------------------------------
    def preprocess(self, text):

        text = str(text)

        text = self.fix_arabic(text)
        text = self.replace_links(text)
        text = self.convert_numbers(text)
        text = self.replace_science(text)
        text = self.replace_emojis(text)
        text = self.replace_shortcuts(text)
        text = self.normalize_laughs(text)
        text = self.remove_keshide(text)
        text = self.fix_typos(text)
        text = self.replace_english(text)
        text = self.replace_finglish(text)

        text = re.sub(r'\s+', ' ', text).strip()
        return text
