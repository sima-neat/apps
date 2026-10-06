"""Answering from the user's documents on /v1/chat/completions (rag_chat.py):
the opt-in field, the question searched with, and where the passages go.
Pure Python: no Flask, no RAG service."""

from __future__ import annotations

import json
import unittest

import rag_chat

# A hit as the RAG service's /search returns it (rag/vectordb.py).
HIT = {
    "content": "The Neat Library is the C++ and Python library for building AI applications.",
    "metadata": {"Header_2": "What Neat Is", "Header_1": "Neat Library Overview"},
    "score": 0.81234,
}


class RagOptionTests(unittest.TestCase):
    def test_the_field_is_removed_before_the_model_server_sees_it(self):
        payload = {"model": "m", "messages": [], "neat_rag": True}
        self.assertEqual(rag_chat.rag_options(payload), {"k": rag_chat.DEFAULT_K})
        self.assertNotIn("neat_rag", payload)

    def test_k_is_bounded_and_bad_values_fall_back(self):
        self.assertEqual(rag_chat.rag_options({"neat_rag": {"k": 50}}), {"k": rag_chat.MAX_K})
        self.assertEqual(rag_chat.rag_options({"neat_rag": {"k": 0}}), {"k": 1})
        self.assertEqual(rag_chat.rag_options({"neat_rag": {"k": "x"}}), {"k": rag_chat.DEFAULT_K})

    def test_absent_false_or_disabled_means_no_documents(self):
        for payload in ({}, {"neat_rag": False}, {"neat_rag": {"enabled": False}}, {"neat_rag": "yes"}):
            self.assertIsNone(rag_chat.rag_options(dict(payload)), payload)


class RagQuestionTests(unittest.TestCase):
    def test_searches_with_the_last_user_text_without_no_think(self):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": [
                {"type": "text", "text": "/no_think What is Neat?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xx"}},
            ]},
        ]
        self.assertEqual(rag_chat.last_user_text(messages), "What is Neat?")
        self.assertEqual(rag_chat.last_user_text([]), "")


class RagPassageTests(unittest.TestCase):
    def test_passages_go_into_the_question_and_the_system_prompt_stays_first(self):
        messages = [{"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "What is Neat?"}]
        out = rag_chat.with_passages(messages, [HIT])
        self.assertEqual([m["role"] for m in out], ["system", "user"],
                         "no second system message: chat templates ignore one mid-conversation")
        self.assertEqual(out[0]["content"], "Be brief.")
        question = out[1]["content"]
        self.assertTrue(question.startswith(rag_chat.INSTRUCTION))
        self.assertIn("[1] Neat Library Overview › What Neat Is", question)
        self.assertIn(HIT["content"], question)
        self.assertTrue(question.endswith("Question: What is Neat?"))
        self.assertEqual(messages[1]["content"], "What is Neat?", "the request's own list is not changed")

    def test_a_picture_question_keeps_its_picture(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,xx"}}]}]
        parts = rag_chat.with_passages(messages, [HIT])[0]["content"]
        self.assertEqual([p["type"] for p in parts], ["text", "image_url"])
        self.assertTrue(parts[0]["text"].endswith("Question: What is this?"))

    def test_no_hits_leaves_the_conversation_as_it_was(self):
        messages = [{"role": "user", "content": "hi"}]
        self.assertEqual(rag_chat.with_passages(messages, []), messages)

    def test_long_passages_are_trimmed(self):
        long_hit = dict(HIT, content="x" * 5000)
        note = rag_chat.with_passages([{"role": "user", "content": "q"}], [long_hit])[0]["content"]
        self.assertLess(len(note), rag_chat.MAX_PASSAGE_CHARS + 400)

    def test_sources_header_is_ascii_json_with_headings_and_scores(self):
        header = rag_chat.sources_header([HIT])
        header.encode("ascii")  # a valid HTTP header value
        self.assertEqual(json.loads(header), [{"source": "", "heading": "Neat Library Overview › What Neat Is", "score": 0.8123}])


class RagEventLoopTests(unittest.TestCase):
    """Building a database on a Flask worker thread (upload, reset) needs an
    event loop there; worker threads have none by default."""

    def test_a_worker_thread_gets_an_event_loop(self):
        import asyncio
        import threading

        from rag.event_loop import ensure_thread_event_loop

        seen = {}

        def worker():
            try:
                asyncio.get_event_loop()
                seen["before"] = "had one"
            except RuntimeError:
                seen["before"] = "none"
            loop = ensure_thread_event_loop()
            seen["after"] = asyncio.get_event_loop() is loop
            seen["again"] = ensure_thread_event_loop() is loop
            loop.close()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertEqual(seen, {"before": "none", "after": True, "again": True})


if __name__ == "__main__":
    unittest.main()
