from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict, Any, Optional
import math


class GlobalFocusEngine:
    def __init__(self, coverage_window_hours: int = 72):
        # Expanded Master Threat Matrix — now includes Swahili, Hausa, and
        # Amharic alongside Bangla/English/Arabic/French/German/Spanish.
        self.critical_keywords = {
            "power_illegal_use": [
                # Bangla
                "বিদ্যুৎ চুরি", "অবৈধ সংযোগ", "অবৈধ বিদ্যুৎ", "অবৈধ বিদ্যুৎ ব্যবহার",
                # English
                "electricity theft", "power theft", "illegal electricity connection", "illegal power use", "energy pilferage",
                # Arabic
                "سرقة الكهرباء", "توصيلات كهربائية غير قانونية", "استخدام غير قانوني للكهرباء",
                # French
                "vol d'électricité", "raccordement électrique illégal", "utilisation illégale d'électricité",
                # German
                "stromdiebstahl", "illegaler stromanschluss", "illegale stromnutzung",
                # Spanish
                "robo de electricidad", "conexión eléctrica ilegal", "uso ilegal de electricidad",
                # Swahili
                "wizi wa umeme", "muunganisho wa umeme haramu", "matumizi haramu ya umeme",
                # Hausa
                "sace wutar lantarki", "hada wutar lantarki ba bisa ka'ida ba",
                # Amharic
                "የኤሌክትሪክ ስርቆት", "ህገወጥ የኤሌክትሪክ ግንኙነት",
            ],
            "food_security_and_hunger": [
                # Bangla
                "খাদ্য সংকট", "দুর্ভিক্ষ", "পুষ্টিহীনতা", "ক্ষুধা", "খাদ্যমূল্য",
                # English
                "food security", "famine", "malnutrition", "hunger", "starvation", "food crisis",
                # Arabic
                "أمن غذائي", "مجاعة", "سوء تغذية", "جوع", "أزمة غذائية",
                # French
                "sécurité alimentaire", "famine", "malnutrition", "faim", "crise alimentaire",
                # German
                "ernährungssicherheit", "hungersnot", "unterernährung", "hunger", "ernährungskrise",
                # Spanish
                "seguridad alimentaria", "hambruna", "desnutrición", "hambre", "crisis alimentaria",
                # Swahili
                "usalama wa chakula", "njaa", "utapiamlo", "baa la njaa", "mgogoro wa chakula",
                # Hausa
                "tsaron abinci", "yunwa", "rashin abinci mai gina jiki", "matsalar abinci",
                # Amharic
                "የምግብ ዋስትና", "ረሃብ", "የተመጣጠነ ምግብ እጥረት", "የምግብ ቀውስ",
            ],
            "economic_and_infrastructure_collapse": [
                # Bangla
                "অর্থনৈতিক সংকট", "মুদ্রাস্ফীতি", "বিদ্যুৎ সংকট", "পানির সংকট", "অবকাঠামো", "ঋণ সংকট",
                # English
                "economic crisis", "inflation", "power outage", "blackout", "water scarcity", "debt distress", "infrastructure collapse",
                # Arabic
                "أزمة اقتصادية", "تضخم", "انقطاع الكهرباء", "شح المياه", "انهيار اقتصادي", "ديون",
                # French
                "crise économique", "inflation", "coupure de courant", "pénurie d'eau", "effondrement économique", "dette",
                # German
                "wirtschaftskrise", "inflation", "stromausfall", "wasserknappheit", "wirtschaftlicher zusammenbruch", "verschuldung",
                # Spanish
                "crisis económica", "inflación", "apagón", "escasez de agua", "colapso económico", "deuda",
                # Swahili
                "mgogoro wa kiuchumi", "mfumko wa bei", "kukatika kwa umeme", "uhaba wa maji", "mzigo wa deni",
                # Hausa
                "matsalar tattalin arziki", "hauhawar farashi", "yankewar wutar lantarki", "karancin ruwa", "bashin da ya wuce kima",
                # Amharic
                "የኢኮኖሚ ቀውስ", "የዋጋ ግሽበት", "የመብራት መቆራረጥ", "የውሃ እጥረት", "የዕዳ ጫና",
            ],
            "public_health_and_epidemic_crisis": [
                # Bangla
                "স্বাস্থ্য সংকট", "মহামারী", "কলেরা", "রোগ ছড়ানো", "চিকিৎসা সংকট",
                # English
                "health crisis", "outbreak", "epidemic", "cholera", "hospital collapse", "disease spread",
                # Arabic
                "أزمة صحية", "تفشي", "وباء", "كوليرا", "انهيار صحي",
                # French
                "crise sanitaire", "épidémie", "choléra", "effondrement de la santé", "propagation de maladie",
                # German
                "gesundheitskrise", "ausbruch", "epidemie", "cholera", "gesundheitskollaps", "krankheitsausbruch",
                # Spanish
                "crisis sanitaria", "brote", "epidemia", "cólera", "colapso de salud", "propagación de enfermedades",
                # Swahili
                "mgogoro wa afya", "mlipuko wa ugonjwa", "kipindupindu", "kuporomoka kwa hospitali", "kuenea kwa ugonjwa",
                # Hausa
                "matsalar lafiya", "barkewar cuta", "kwalara", "rugujewar asibiti", "yaduwar cuta",
                # Amharic
                "የጤና ቀውስ", "ወረርሽኝ", "ኮሌራ", "የሆስፒታል ውድቀት", "የበሽታ ስርጭት",
            ],
            "disaster_and_nature": [
                # Bangla
                "ভূমিকম্প", "বন্যা", "ঘূর্ণিঝড়", "জলোচ্ছ্বাস", "বিপৎসীমা",
                # English
                "earthquake", "flood", "hurricane", "tsunami", "disaster",
                # Arabic
                "زلزال", "فيضان", "إعصار", "تسونامي", "كارثة",
                # French
                "tremblement de terre", "inondation", "ouragan", "tsunami", "catastrophe",
                # German
                "erdbeben", "überschwemmung", "hurrikan", "tsunami", "katastrophe",
                # Spanish
                "terremoto", "inundación", "huracán", "tsunami", "desastre",
                # Swahili
                "tetemeko la ardhi", "mafuriko", "kimbunga", "tsunami", "maafa",
                # Hausa
                "girgizar kasa", "ambaliyar ruwa", "guguwa", "tsunami", "bala'i",
                # Amharic
                "የመሬት መንቀጥቀጥ", "ጎርፍ", "አውሎ ንፋስ", "ጸናፍ", "አደጋ",
            ],
            "security_and_conflict": [
                # Bangla
                "হামলা", "আগুন", "বিস্ফোরণ", "নিহত", "গুলি", "অগ্নিকাণ্ড",
                # English
                "attack", "fire", "explosion", "killed", "shooting", "terror",
                # Arabic
                "هجوم", "حريق", "انفجار", "قتلى", "إطلاق نار", "إرهاب",
                # French
                "attaque", "incendie", "explosion", "morts", "fusillade", "terrorisme",
                # German
                "angriff", "feuer", "explosion", "getötet", "schiesserei", "terror",
                # Spanish
                "ataque", "incendio", "explosión", "muertos", "tiroteo", "terrorismo",
                # Swahili
                "shambulio", "moto", "mlipuko", "kuuawa", "risasi", "ugaidi",
                # Hausa
                "hari", "wuta", "fashewa", "kisa", "harbi", "ta'addanci",
                # Amharic
                "ጥቃት", "እሳት", "ፍንዳታ", "ተገደለ", "ተኩስ", "ሽብርተኝነት",
            ],
            "urgency_boosters": [
                # Bangla
                "জরুরি", "সতর্কতা", "ভয়াবহ", "তাত্ক্ষণিক",
                # English
                "emergency", "alert", "warning", "critical", "immediate",
                # Arabic
                "طوارئ", "إنذار", "تحذير", "حرج", "فوري",
                # French
                "urgence", "alerte", "avertissement", "critique", "immédiat",
                # German
                "notfall", "alarm", "warnung", "kritisch", "sofortig",
                # Spanish
                "emergencia", "alerta", "advertencia", "crítico", "inmediato",
                # Swahili
                "dharura", "tahadhari", "onyo", "hatari", "haraka",
                # Hausa
                "gaggawa", "faɗakarwa", "gargaɗi", "mahimmanci", "nan take",
                # Amharic
                "አስቸኳይ", "ማንቂያ", "ማስጠንቀቂያ", "ወሳኝ", "አፋጣኝ",
            ]
        }

        self.top_10_by_language = {
            "bn": [], "en": [], "es": [], "de": [], "fr": [], "ar": [],
            "sw": [],  # Swahili
            "ha": [],  # Hausa
            "am": [],  # Amharic
        }

        # --- Coverage tracking for the "ignored but urgent" signal ---
        # For each category, keep a rolling deque of timestamps of every
        # article that matched it. Coverage count = how many articles
        # mentioned this category within the window. Low count = ignored.
        self.coverage_window = timedelta(hours=coverage_window_hours)
        self.category_mentions: Dict[str, deque] = {
            cat: deque() for cat in self.critical_keywords
        }

    def _record_coverage(self, matched_categories: List[str], now: datetime):
        """Log this article's matched categories into the rolling coverage
        window, and prune anything older than the window."""
        for cat in matched_categories:
            dq = self.category_mentions[cat]
            dq.append(now)
            cutoff = now - self.coverage_window
            while dq and dq[0] < cutoff:
                dq.popleft()

    def _ignored_multiplier(self, matched_categories: List[str]) -> float:
        """Inverse-frequency boost: categories with few recent mentions
        across ALL sources/languages get amplified; categories already
        flooding the system get dampened. Returns a multiplier, not a
        replacement for severity — combine with the raw score."""
        if not matched_categories:
            return 1.0

        counts = [len(self.category_mentions[cat]) for cat in matched_categories]
        avg_count = sum(counts) / len(counts)

        # log-scaled inverse frequency: 1 mention -> high boost,
        # 100+ mentions -> multiplier settles near 1 (no boost, no penalty
        # below 1 so a real crisis is never scored below its raw severity)
        boost = 1.0 + (2.0 / math.log(avg_count + 2.0))
        return round(boost, 3)

    def evaluate_article(self, article_id: str, title: str, text: str, language: str):
        """Scans article for threat categories, computes a severity score,
        then applies an 'ignored crisis' boost based on how little
        coverage this category has had recently across all languages."""
        if language not in self.top_10_by_language:
            language = "en"

        combined_text = f"{title} {text}".lower()
        severity_score = 0
        matched_categories = []

        for category, keywords in self.critical_keywords.items():
            for word in keywords:
                if word in combined_text:
                    weight = 45 if category in [
                        "power_illegal_use", "food_security_and_hunger",
                        "economic_and_infrastructure_collapse",
                        "public_health_and_epidemic_crisis",
                    ] else 35
                    severity_score += weight
                    if category not in matched_categories:
                        matched_categories.append(category)

        if len(matched_categories) >= 2:
            severity_score += 35

        if severity_score == 0:
            return

        now = datetime.now()

        # Compute the ignored-crisis boost BEFORE recording this article's
        # own mentions, so it's scored against prior coverage, not itself.
        ignored_multiplier = self._ignored_multiplier(matched_categories)
        self._record_coverage(matched_categories, now)

        display_severity = min(severity_score, 100)  # 0-100 scale, for showing to a user
        # Ranking score is intentionally UNCAPPED: capping here would tie
        # multiple high-severity articles together and let the sort fall
        # back to timestamp, silently erasing the ignored-crisis signal.
        ranking_score = round(severity_score * ignored_multiplier, 2)

        article_entry = {
            "id": article_id,
            "title": title,
            "text": text,
            "language_detected": language,
            "severity_score": display_severity,
            "ignored_multiplier": ignored_multiplier,
            "concern_score": ranking_score,
            "threat_categories": matched_categories,
            "timestamp": now.isoformat(),
        }

        lang_list = self.top_10_by_language[language]
        existing_idx = next((i for i, item in enumerate(lang_list) if item["id"] == article_id), -1)
        if existing_idx != -1:
            lang_list[existing_idx] = article_entry
        else:
            lang_list.append(article_entry)

        lang_list.sort(key=lambda x: x["concern_score"], reverse=True)
        self.top_10_by_language[language] = lang_list[:10]

    def get_all_top_focus(self):
        """Returns the ranked Top 10 concerning-but-often-ignored articles
        across all languages, filtering out entries older than 24h."""
        now = datetime.now()
        result = {}
        for lang, articles in self.top_10_by_language.items():
            valid_articles = []
            for article in articles:
                article_time = datetime.fromisoformat(article["timestamp"])
                age_in_hours = (now - article_time).total_seconds() / 3600
                if age_in_hours <= 24:
                    valid_articles.append(article)
            self.top_10_by_language[lang] = valid_articles
            result[lang] = valid_articles
        return result

    def get_category_coverage_snapshot(self) -> Dict[str, int]:
        """Debug/insight helper: how many articles have hit each category
        in the current rolling window right now. Low numbers here are
        exactly the categories the 'ignored' boost is amplifying."""
        return {cat: len(dq) for cat, dq in self.category_mentions.items()}


global_focus_engine = GlobalFocusEngine()