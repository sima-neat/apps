"""Answering from the user's documents on /v1/chat/completions (rag_chat.py):
the opt-in field, the question searched with, and where the passages go.
Pure Python: no Flask, no RAG service."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
        for huge in (float("inf"), float("-inf"), float("nan")):   # JSON 1e999 decodes to inf
            self.assertEqual(rag_chat.rag_options({"neat_rag": {"k": huge}}), {"k": rag_chat.DEFAULT_K})

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
        long = "x" * 70000 + " What is Neat?"
        self.assertEqual(len(rag_chat.last_user_text([{"role": "user", "content": long}])), rag_chat.MAX_QUERY_CHARS)
        self.assertTrue(rag_chat.last_user_text([{"role": "user", "content": long}]).endswith("What is Neat?"))



class ChatRequestShapeTests(unittest.TestCase):
    """The chat route answers 400 with chat_request_error's reason, before
    anything reads the request, instead of failing later with a 500."""

    def test_sound_requests_pass(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
        for payload in (
            {"messages": [{"role": "user", "content": "hi"}]},
            {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": [{"type": "text", "text": "q"}, image]}]},
            {"messages": [{"role": "assistant", "content": None}, {"role": "user", "content": "q"}], "neat_rag": True},
        ):
            self.assertIsNone(rag_chat.chat_request_error(payload), payload)

    def test_every_malformed_shape_has_a_reason(self):
        bad_messages = (None, 1, True, 1.5, "hi", {}, [], ["hi"], [None], [{"content": "x"}], [{"role": 1, "content": "x"}],
                        [{"role": "user"}], [{"role": "user", "content": None}], [{"role": "user", "content": 5}],
                        [{"role": "user", "content": [5]}], [{"role": "user", "content": [{"text": "x"}]}],
                        [{"role": "user", "content": [{"type": "text", "text": 5}]}],
                        [{"role": "user", "content": [{"type": "image_url", "image_url": "data:x"}]}],
                        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": 5}}]}])
        for messages in bad_messages:
            with self.subTest(messages=messages):
                message, param = rag_chat.chat_request_error({"messages": messages})
                self.assertEqual(param, "messages")
                self.assertTrue(message)
        for body in ([], "x", 1, None):
            with self.subTest(body=body):
                self.assertEqual(rag_chat.chat_request_error(body)[1], None)


class RagContextBudgetTests(unittest.TestCase):
    def test_passages_together_fit_the_budget_and_only_those_are_used(self):
        hits = [{"content": "x" * 1300, "metadata": {"Header_1": f"Part {i}"}} for i in range(10)]
        kept = rag_chat.fit_passages(hits)
        self.assertLessEqual(sum(len(h["content"]) for h in kept), rag_chat.MAX_CONTEXT_CHARS + 1)
        self.assertEqual(len(kept), 3)          # two whole passages and a trimmed third
        self.assertTrue(kept[-1]["content"].endswith("…"))
        self.assertEqual(json.loads(rag_chat.sources_header(kept))[-1]["heading"], "Part 2")

    def test_small_passages_all_fit_and_empty_ones_are_skipped(self):
        hits = [{"content": "short"}, {"content": "  "}, {"content": "also short"}]
        self.assertEqual([h["content"] for h in rag_chat.fit_passages(hits)], ["short", "also short"])


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
        self.assertIn("Information:\n" + HIT["content"], question)
        self.assertNotIn("What Neat Is", question, "headings stay out of the prompt; small models copied them")
        self.assertNotIn("[1]", question)
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
        header = rag_chat.sources_header([HIT], "neat.md")
        header.encode("ascii")  # a valid HTTP header value
        self.assertEqual(json.loads(header), [{"source": "neat.md", "heading": "Neat Library Overview › What Neat Is",
                                               "score": 0.8123}])

    def test_the_source_is_the_document_the_database_was_built_from(self):
        self.assertEqual(rag_chat.document_name({"input": "/data/apps/src/common/rag/neat.md"}), "neat.md")
        self.assertEqual(rag_chat.document_name({"input": "guide.md"}), "guide.md")
        self.assertEqual(rag_chat.document_name({}), "")
        self.assertEqual(rag_chat.passage_source(HIT)["source"], "")
        named = dict(HIT, metadata=dict(HIT["metadata"], source="other.md"))
        self.assertEqual(rag_chat.passage_source(named, "neat.md")["source"], "other.md",
                         "a passage that names its file keeps it")

    def test_long_headings_and_many_sources_fit_one_header(self):
        long_hit = dict(HIT, metadata={"Header_1": "ü" * 500})
        self.assertEqual(len(rag_chat.passage_source(long_hit)["heading"]), rag_chat.MAX_HEADING_CHARS)
        header = rag_chat.sources_header([long_hit] * rag_chat.MAX_K, "x" * 200 + ".md")
        header.encode("ascii")
        self.assertLessEqual(len(header), rag_chat.MAX_SOURCES_HEADER_BYTES)
        sources = json.loads(header)
        self.assertTrue(0 < len(sources) < rag_chat.MAX_K, "trailing entries are dropped, the first ones kept")
        self.assertEqual(sources[0]["heading"], "ü" * rag_chat.MAX_HEADING_CHARS)

    def test_a_malformed_content_part_does_not_break_the_passages(self):
        messages = [{"role": "user", "content": ["stray", {"type": "text", "text": "q"}]}]
        parts = rag_chat.with_passages(messages, [HIT])[0]["content"]
        self.assertEqual(parts[0], "stray")
        self.assertTrue(parts[1]["text"].endswith("Question: q"))


class RagSearchFailedTests(unittest.TestCase):
    def test_cleared_documents_ask_for_an_upload_or_reset(self):
        message, status = rag_chat.search_failed_reply(database_exists=False)
        self.assertEqual(status, 409)
        self.assertIn("cleared", message)
        self.assertIn("Upload a Markdown file or reset", message)
        self.assertNotIn("Try again", message)

    def test_a_built_database_whose_service_is_down_says_try_again(self):
        message, status = rag_chat.search_failed_reply(database_exists=True)
        self.assertEqual(status, 503)
        self.assertIn("Try again in a moment", message)

    def test_a_service_that_exited_says_waiting_wont_help(self):
        message, status = rag_chat.search_failed_reply(database_exists=True, service_failed=True)
        self.assertEqual(status, 503)
        self.assertIn("stopped", message)
        self.assertNotIn("Try again in a moment", message)


class RagServedDatabaseTests(unittest.TestCase):
    """The Studio searches only a service that serves its own database, not
    one the CLI started on the same port with `/rag db`."""

    def test_the_same_file_is_not_another_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "milvus.db"
            self.assertIsNone(rag_chat.other_database(str(db), str(db)))
            self.assertIsNone(rag_chat.other_database(f"{tmp}/./milvus.db", str(db)))

    def test_a_different_file_is_reported(self):
        self.assertEqual(rag_chat.other_database("/data/cli/milvus.db", "/opt/studio/milvus.db"),
                         "/data/cli/milvus.db")

    def test_no_service_or_a_loading_one_is_left_to_the_search(self):
        self.assertIsNone(rag_chat.other_database(None, "/opt/studio/milvus.db"))

    def test_a_service_that_does_not_name_its_database_is_refused(self):
        self.assertEqual(rag_chat.other_database("", "/opt/studio/milvus.db"), rag_chat.UNNAMED_DATABASE)

    def test_every_studio_rag_call_goes_through_the_checked_client(self):
        source = (Path(__file__).resolve().parents[2] / "src/python/ui/flask_app.py").read_text(encoding="utf-8")
        for target in ("ensure_rag_modules_loaded()", "rag_db_client"):
            for call in (".search(", ".is_server_up(", ".list_documents("):
                self.assertNotIn(target + call, source)
        self.assertGreaterEqual(source.count("checked_rag_client()"), 6)
        # Each search answers the mismatch as JSON instead of an HTML 500.
        searches = source.count("checked_rag_client().search(")
        self.assertEqual(searches, 3)
        self.assertEqual(source.count("except RagServiceMismatch"), searches + 2)  # + /rag/status, /rag/inspect

    def test_the_standalone_page_shows_a_failed_request_in_the_chat(self):
        page = (Path(__file__).resolve().parents[2] / "src/python/ui/static/newui.js").read_text(encoding="utf-8")
        start = page.index("const response = await fetch('/upload', {")
        handler = page[page.index("} catch (error) {", start):page.index("};", page.index("} catch (error) {", start))]
        self.assertIn("addChatMessage(", handler)
        self.assertIn("hideAbortButton()", handler)


class RagServiceIdentityTests(unittest.TestCase):
    """What vectordb.served_database() makes of a RAG service's ``/`` page.
    Taken from the source so the test does not import the service's stack."""

    @classmethod
    def setUpClass(cls):
        import ast
        from typing import Any, Optional
        path = Path(__file__).resolve().parents[2] / "src/python/rag/vectordb.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "served_database")
        namespace = {"Any": Any, "Optional": Optional}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
        cls.served = staticmethod(namespace["served_database"])

    def test_a_named_database(self):
        self.assertEqual(self.served({"service": "rag-vectordb", "db_path": "/x/milvus.db"}), "/x/milvus.db")

    def test_a_loading_service_is_left_to_the_search(self):
        self.assertIsNone(self.served({"service": "rag-vectordb", "db_path": None}))

    def test_a_reply_that_does_not_name_a_database_is_refused(self):
        for reply in (None, ["list"], {"service": "rag-vectordb"}, {"db_path": "/x/milvus.db"},
                      {"service": "other", "db_path": "/x/milvus.db"}, {"service": "rag-vectordb", "db_path": ""},
                      {"service": "rag-vectordb", "db_path": 3}):
            self.assertEqual(self.served(reply), "", reply)


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

    def test_the_route_closes_the_loop_it_created(self):
        import asyncio
        import threading

        from rag.event_loop import close_thread_event_loop, ensure_thread_event_loop

        seen = {}

        def worker():
            loop = ensure_thread_event_loop()
            close_thread_event_loop()
            seen["closed"] = loop.is_closed()
            try:
                asyncio.get_event_loop()
                seen["after"] = "had one"
            except RuntimeError:
                seen["after"] = "none"
            close_thread_event_loop()   # nothing created: no error

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertEqual(seen, {"closed": True, "after": "none"})


if __name__ == "__main__":
    unittest.main()
