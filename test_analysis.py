import unittest

from analysis import ERROR_CATEGORIES, analyze_words, build_features


class AnalyzeWordsTests(unittest.TestCase):
    def test_detects_omission(self):
        findings = analyze_words("gato pula muro", "gato pula o muro")
        self.assertEqual(findings[0]["type"], "Revisão manual (fora das regras dos slides)")

    def test_detects_substitution(self):
        findings = analyze_words("caza", "casa")
        self.assertEqual(findings[0]["type"], ERROR_CATEGORIES[0])

    def test_detects_addition(self):
        findings = analyze_words("gato pula o muro", "gato pula muro")
        self.assertEqual(findings[0]["type"], "Revisão manual (fora das regras dos slides)")

    def test_detects_accentuation(self):
        findings = analyze_words("voce", "você")
        self.assertEqual(findings[0]["type"], "Revisão manual (fora das regras dos slides)")

    def test_builds_ten_classifier_features(self):
        findings = analyze_words("caza", "casa")
        self.assertEqual(build_features(["caza"], findings), [1] + [0] * 9)

    def test_classifies_examples_from_all_ten_slide_categories(self):
        examples = (
            ("traveceiro", "travesseiro"),
            ("girasou", "girassol"),
            ("mamã", "mamão"),
            ("derepente", "de repente"),
            ("gostarão", "gostaram"),
            ("chapel", "chapéu"),
            ("machugado", "machucado"),
            ("carata", "carta"),
            ("núsica", "música"),
            ("braco", "barco"),
        )
        for index, (transcription, reference) in enumerate(examples):
            with self.subTest(category=ERROR_CATEGORIES[index]):
                finding = analyze_words(transcription, reference)[0]
                self.assertEqual(finding["type"], ERROR_CATEGORIES[index])

    def test_visual_categories_require_image_review(self):
        for transcription, reference in (("núsica", "música"), ("braco", "barco")):
            with self.subTest(transcription=transcription):
                finding = analyze_words(transcription, reference)[0]
                self.assertTrue(finding["requires_visual_review"])


if __name__ == "__main__":
    unittest.main()