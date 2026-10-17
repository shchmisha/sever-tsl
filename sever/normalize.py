"""Deterministic normalization. Never repairs a guessed character or organization."""
import re
from datetime import date
import unicodedata

def text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()

def normalize(field: str, value: str) -> str | None:
    if field.endswith("number"):
        value = value.strip().removeprefix("№").strip()
    value = text(value)
    if field.endswith("date"):
        match = re.search(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?!\d)", value)
        iso = re.search(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)", value)
        months = {"января":1,"февраля":2,"марта":3,"апреля":4,"мая":5,"июня":6,"июля":7,"августа":8,"сентября":9,"октября":10,"ноября":11,"декабря":12}
        words = re.search(r"(?<!\d)(\d{1,2})\s+("+"|".join(months)+r")\s+(\d{4})(?!\d)", value.casefold())
        try:
            if iso:
                return date(*map(int, iso.groups())).isoformat()
            if match:
                d, m, y = map(int, match.groups())
                return date(y, m, d).isoformat()
            if words:
                d, month, y = words.groups()
                return date(int(y), months[month], int(d)).isoformat()
        except ValueError:
            pass
        return None
    if field.endswith("number"):
        return re.sub(r"\s+", "", value.removeprefix("№")).replace("–", "-").replace("—", "-") or None
    return value or None

def comparison(field: str, value: str | None) -> str | None:
    """Evaluation-only formatting equivalence; meaningful words/digits are retained."""
    if value is None:
        return None
    result = normalize(field, value)
    if result is None:
        return None
    result = result.casefold().replace("ё", "е").translate(str.maketrans("«»“”", '\"\"\"\"'))
    if field in {"supplier", "consignor", "consignee", "customer"}:
        # Canonicalize only an explicit legal-form prefix, preserving name words.
        for full, short in (
            ("общество с ограниченной ответственностью", "ооо"),
            ("публичное акционерное общество", "пао"),
            ("закрытое акционерное общество", "зао"),
            ("открытое акционерное общество", "оао"),
            ("акционерное общество", "ао"),
            ("индивидуальный предприниматель", "ип"),
        ):
            result = re.sub(r"^" + full + r"\b", short, result)
        result = result.replace('"', "")
        result = re.sub(r"(?<!\d)[.,;:]|[.,;:](?!\d)", " ", result)
        result = re.sub(r"\s*([-+&])\s*", r"\1", result.replace("–", "-").replace("—", "-"))
    elif field.endswith("address"):
        result = result.replace('"', "")
        # Equivalent printed numero glyphs; retain every house/unit digit.
        result = re.sub(r"(?:№|\bno(?=\s*\d))", "№", result)
        # Punctuation in ул.Софийская/д.6 is formatting; a numeric 16.1 stays 16.1.
        result = re.sub(r"(?<!\d)[.,;:]|[.,;:](?!\d)", " ", result)
        result = re.sub(r"\s*-\s*", "-", result.replace("–", "-").replace("—", "-"))
    return text(result)
