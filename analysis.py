import re
import unicodedata
from difflib import SequenceMatcher


ERROR_CATEGORIES = (
    "1. Representações múltiplas",
    "2. Apoio na oralidade",
    "3. Omissões de letras ou sílabas",
    "4. Junção ou separação indevida",
    "5. Confusão AM x ÃO",
    "6. Generalização",
    "7. Trocas surdas-sonoras",
    "8. Acréscimos de letras",
    "9. Letras parecidas",
    "10. Inversões",
)
ERROR_CATEGORY_DESCRIPTIONS = (
    "Grafias diferentes para sons iguais, como ss/c, x/ch e s/ç.",
    "Escrita guiada pela pronúncia, como l final escrito como u.",
    "Ausência de letras ou partes da palavra esperada.",
    "Palavras unidas ou separadas indevidamente.",
    "Troca entre terminações am e ão.",
    "Aplicação de uma regra conhecida a uma palavra em que ela não se aplica.",
    "Troca entre consoantes surdas e sonoras, como p/b ou t/d.",
    "Inclusão de letras que não aparecem na palavra esperada.",
    "Troca de letras com formas gráficas semelhantes; exige conferência da imagem.",
    "Inversão na posição ou orientação de letras; exige conferência da imagem.",
)

MANUAL_REVIEW = "Revisão manual (fora das regras dos slides)"
VOICING_PAIRS = {("p", "b"), ("t", "d"), ("c", "g"), ("qu", "gu"), ("f", "v"), ("s", "z"), ("x", "j"), ("ch", "j")}
SIMILAR_LETTER_PAIRS = {("m", "n"), ("n", "u"), ("t", "f"), ("i", "l"), ("c", "e"), ("g", "q"), ("h", "n"), ("cl", "ch"), ("lh", "nh")}


def _tokens(text):
    return re.findall(r"[\wÀ-ÿ]+(?:[-'][\wÀ-ÿ]+)*", text.lower(), flags=re.UNICODE)


def _without_accents(word):
    return "".join(character for character in unicodedata.normalize("NFD", word) if unicodedata.category(character) != "Mn")


def _phonetic_signature(word):
    word = _without_accents(word)
    word = word.replace("ch", "§").replace("x", "§")
    word = word.replace("ss", "s").replace("ç", "s")
    word = re.sub(r"c(?=[ei])", "s", word)
    word = re.sub(r"(?<=[aeiou])s(?=[aeiou])", "z", word)
    word = word.replace("§", "sh")
    word = re.sub(r"[gj](?=[ei])", "j", word)
    return word


def _is_orality_variant(expected, detected):
    expected_clean = _without_accents(expected)
    detected_clean = _without_accents(detected)
    if expected_clean == detected_clean:
        return False
    bases = {expected_clean, expected_clean.replace("ss", "s")}
    variants = set()
    for base in bases:
        variants.update({
            re.sub(r"l$", "u", base),
            re.sub(r"l(?=[bcdfghjklmnpqrstvwxyz])", "u", base),
            re.sub(r"^en", "in", base),
            base.replace("enx", "inch"),
            re.sub(r"que$", "qui", base),
        })
    return detected_clean in variants


def _is_am_ao_confusion(expected, detected):
    expected_clean = _without_accents(expected)
    detected_clean = _without_accents(detected)
    for expected_ending, detected_ending in (("am", "ao"), ("ao", "am")):
        if expected_clean.endswith(expected_ending) and detected_clean == expected_clean[:-2] + detected_ending:
            return True
    return False


def _is_generalization(expected, detected):
    expected_clean = _without_accents(expected)
    detected_clean = _without_accents(detected)
    return expected_clean.endswith("eu") and detected_clean == expected_clean[:-2] + "el"


def _has_character_pair(expected, detected, pairs):
    expected_clean = _without_accents(expected)
    detected_clean = _without_accents(detected)
    for operation, start_expected, end_expected, start_detected, end_detected in SequenceMatcher(
        a=expected_clean, b=detected_clean, autojunk=False
    ).get_opcodes():
        if operation != "replace":
            continue
        old = expected_clean[start_expected:end_expected]
        new = detected_clean[start_detected:end_detected]
        if (old, new) in pairs or (new, old) in pairs:
            return True
    return False


def _is_inversion(expected, detected):
    expected_clean = _without_accents(expected)
    detected_clean = _without_accents(detected)
    if expected_clean == detected_clean:
        return False
    if {expected_clean, detected_clean} in ({"b", "d"}, {"p", "q"}):
        return True
    if len(expected_clean) != len(detected_clean) or sorted(expected_clean) != sorted(detected_clean):
        return False
    differences = [index for index, (left, right) in enumerate(zip(expected_clean, detected_clean)) if left != right]
    return len(differences) == 2 and differences[1] == differences[0] + 1


def _classify_word(expected, detected):
    if _without_accents(expected) == _without_accents(detected):
        return MANUAL_REVIEW, False
    if _is_am_ao_confusion(expected, detected):
        return ERROR_CATEGORIES[4], False
    if _is_generalization(expected, detected):
        return ERROR_CATEGORIES[5], False
    if _is_orality_variant(expected, detected):
        return ERROR_CATEGORIES[1], False
    if _is_inversion(expected, detected):
        return ERROR_CATEGORIES[9], True
    if _phonetic_signature(expected) == _phonetic_signature(detected):
        return ERROR_CATEGORIES[0], False
    if _has_character_pair(expected, detected, VOICING_PAIRS):
        return ERROR_CATEGORIES[6], False
    if _has_character_pair(expected, detected, SIMILAR_LETTER_PAIRS):
        return ERROR_CATEGORIES[8], True

    operations = [opcode[0] for opcode in SequenceMatcher(a=expected, b=detected, autojunk=False).get_opcodes() if opcode[0] != "equal"]
    if operations and set(operations) == {"delete"}:
        return ERROR_CATEGORIES[2], False
    if operations and set(operations) == {"insert"}:
        return ERROR_CATEGORIES[7], False
    return MANUAL_REVIEW, False


def _segmentation_match(expected_words, detected_words):
    for total_words in range(3, 7):
        for expected_count in range(1, min(3, len(expected_words)) + 1):
            detected_count = total_words - expected_count
            if detected_count < 1 or detected_count > 3 or detected_count > len(detected_words):
                continue
            if expected_count == detected_count:
                continue
            expected_text = "".join(expected_words[:expected_count])
            detected_text = "".join(detected_words[:detected_count])
            if expected_text == detected_text:
                return expected_count, detected_count
    return None


def _finding(expected, detected, category, requires_visual_review=False):
    return {
        "type": category,
        "categories": [category] if category in ERROR_CATEGORIES else [],
        "expected": expected,
        "detected": detected,
        "requires_visual_review": requires_visual_review,
    }


def analyze_words(transcription, reference=""):
    detected = _tokens(transcription)
    expected = _tokens(reference)
    if not expected:
        return []

    findings = []
    for operation, start_expected, end_expected, start_detected, end_detected in SequenceMatcher(a=expected, b=detected, autojunk=False).get_opcodes():
        if operation == "equal":
            continue
        expected_group = expected[start_expected:end_expected]
        detected_group = detected[start_detected:end_detected]
        while expected_group and detected_group:
            segmentation = _segmentation_match(expected_group, detected_group)
            if segmentation:
                expected_count, detected_count = segmentation
                findings.append(_finding(
                    " ".join(expected_group[:expected_count]),
                    " ".join(detected_group[:detected_count]),
                    ERROR_CATEGORIES[3],
                ))
                expected_group = expected_group[expected_count:]
                detected_group = detected_group[detected_count:]
                continue

            target = expected_group.pop(0)
            observed = detected_group.pop(0)
            category, visual_review = _classify_word(target, observed)
            findings.append(_finding(target, observed, category, visual_review))

        findings.extend(_finding(word, "", MANUAL_REVIEW) for word in expected_group)
        findings.extend(_finding("", word, MANUAL_REVIEW) for word in detected_group)
    return findings


def build_features(words, findings):
    counts = {category: 0 for category in ERROR_CATEGORIES}
    for finding in findings:
        for category in finding.get("categories", []):
            if category in counts:
                counts[category] += 1
    return [counts[category] for category in ERROR_CATEGORIES]