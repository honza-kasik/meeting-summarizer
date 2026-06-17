import copy
import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyzer = _load_module("analyze_meeting_topics", "analyzer/analyze_meeting_topics.py")
generator = _load_module("generate_meeting_article", "article-generator/generate_meeting_article.py")


class ArticleBriefGenerationTests(unittest.TestCase):
    def test_build_llm_query_payload_is_chronological_and_rich(self):
        topics = [
            {
                "topic_id": 10,
                "time_minutes": 26.4,
                "time_range": "5.0–31.4 min",
                "start_minute": 5.0,
                "end_minute": 31.4,
                "topic_type": "discussion",
                "topic_hint": "průběh stavby",
                "speaker_count": 5,
                "dominant_speaker_ratio": 0.41,
                "top_lemmas": ["stavba", "silnice", "harmonogram", "Olomoucká", "výkop", "obyvatel"],
                "segments_count": 6,
                "discussion_intensity": "high",
                "topic_summary_hint": "průběh stavby; klíčová slova: stavba, silnice, harmonogram; typy zmínek: concern, decision",
                "representative_evidence": [
                    {"text": f"Stavba evidence {i}", "time_range": "5.0–10.0 min", "start_minute": 5.0 + i, "end_minute": 6.0 + i, "evidence_type": "discussion"}
                    for i in range(1, 7)
                ],
            },
            {
                "topic_id": 11,
                "time_minutes": 14.2,
                "time_range": "40.0–54.2 min",
                "start_minute": 40.0,
                "end_minute": 54.2,
                "topic_type": "discussion",
                "topic_hint": "dotace a financování",
                "speaker_count": 4,
                "dominant_speaker_ratio": 0.52,
                "top_lemmas": ["dotace", "škola", "rozpočet", "projekt", "korun"],
                "segments_count": 3,
                "discussion_intensity": "medium",
                "topic_summary_hint": "dotace a financování; klíčová slova: dotace, škola, rozpočet; typy zmínek: decision",
                "representative_evidence": [
                    {"text": f"Dotace evidence {i}", "time_range": "40.0–45.0 min", "start_minute": 40.0 + i, "end_minute": 41.0 + i, "evidence_type": "decision"}
                    for i in range(1, 6)
                ],
            },
            {
                "topic_id": 12,
                "time_minutes": 5.1,
                "time_range": "80.0–85.1 min",
                "start_minute": 80.0,
                "end_minute": 85.1,
                "topic_type": "procedural",
                "topic_hint": "postup orgánů města",
                "speaker_count": 2,
                "dominant_speaker_ratio": 0.88,
                "top_lemmas": ["program", "usnesení", "zápis"],
                "segments_count": 1,
                "discussion_intensity": "low",
                "topic_summary_hint": "postup orgánů města; klíčová slova: program, usnesení, zápis; typy zmínek: procedural",
                "representative_evidence": [
                    {"text": f"Procedural evidence {i}", "time_range": "80.0–85.0 min", "start_minute": 80.0 + i, "end_minute": 81.0 + i, "evidence_type": "procedural"}
                    for i in range(1, 5)
                ],
            },
            {
                "topic_id": 13,
                "time_minutes": 2.4,
                "time_range": "90.0–92.4 min",
                "start_minute": 90.0,
                "end_minute": 92.4,
                "topic_type": "procedural",
                "topic_hint": "postup orgánů města",
                "speaker_count": 2,
                "dominant_speaker_ratio": 0.9,
                "top_lemmas": ["bod", "program"],
                "segments_count": 1,
                "discussion_intensity": "low",
                "topic_summary_hint": "krátký bod",
                "representative_evidence": [{"text": "Ignore me", "time_range": "90.0–92.4 min", "start_minute": 90.0, "end_minute": 92.4, "evidence_type": "procedural"}],
            },
        ]

        brief = analyzer.build_llm_query_payload(topics, min_minutes=3.0, max_topics=3)

        self.assertEqual(list(brief.keys()), ["meeting_overview", "priority_topics", "topics"])
        self.assertEqual([topic["order"] for topic in brief["topics"]], [1, 2, 3])
        self.assertEqual([topic["start_minute"] for topic in brief["topics"]], [5.0, 40.0, 80.0])
        self.assertEqual(
            [topic["topic_hint"] for topic in brief["priority_topics"]],
            ["průběh stavby", "dotace a financování", "postup orgánů města"],
        )
        self.assertEqual(len(brief["topics"][0]["evidence"]), 5)
        self.assertEqual(len(brief["topics"][1]["evidence"]), 4)
        self.assertEqual(len(brief["topics"][2]["evidence"]), 3)
        self.assertEqual(
            brief["topics"][0]["keywords"],
            ["stavba", "silnice", "harmonogram", "Olomoucká", "výkop", "obyvatel"],
        )
        self.assertEqual(brief["topics"][1]["speaker_count"], 4)
        self.assertEqual(brief["meeting_overview"]["included_topic_count"], 3)
        self.assertEqual(brief["meeting_overview"]["meeting_character"], "dominated_by_one_topic")
        self.assertEqual(brief["meeting_overview"]["procedural_share"], 0.11)
        self.assertEqual(brief["meeting_overview"]["discussion_share"], 0.89)

    def test_build_llm_prompt_contains_new_contract_and_guidance(self):
        brief = json.loads((REPO_ROOT / "tests/fixtures/long_meeting_brief.json").read_text(encoding="utf-8"))

        prompt = generator.build_llm_prompt(brief)

        self.assertIn("Writing Objective:", prompt)
        self.assertIn("Reader Value:", prompt)
        self.assertIn("700 až 1100 slov", prompt)
        self.assertIn("zastupitelé dále řešili různé body", prompt)
        self.assertIn("SUMMARY:", prompt)
        self.assertIn("ARTICLE_BODY:", prompt)
        self.assertIn("Priority Topics:", prompt)
        self.assertIn("Topics In Reading Order:", prompt)

    def test_fixture_prompt_exposes_richer_brief_shape(self):
        brief = json.loads((REPO_ROOT / "tests/fixtures/long_meeting_brief.json").read_text(encoding="utf-8"))

        normalized = generator.prepare_topics_for_llm(copy.deepcopy(brief))
        prompt = generator.build_llm_prompt(normalized)

        self.assertIn('"meeting_character": "mixed_focus"', prompt)
        self.assertIn('"top_3_longest_topics"', prompt)
        self.assertIn('"speaker_count": 6', prompt)
        self.assertIn('"keywords": [', prompt)
        self.assertIn('"discussion_intensity": "high"', prompt)
        self.assertIn('"topic_summary_hint"', prompt)
        self.assertIn('"evidence_type": "decision"', prompt)


if __name__ == "__main__":
    unittest.main()
