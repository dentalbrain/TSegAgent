"""Flex-tier handling in fdi_classifier: retry schedule, typed error, no silent tier switch."""

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import fdi_classifier  # noqa: E402


class FakeRateLimitError(Exception):
    """Shape of openai.RateLimitError for a flex capacity refusal (matched by name/status/message)."""

    status_code = 429

    def __init__(self, message="Error code: 429 - {'error': {'message': 'Resource Unavailable', "
                               "'type': 'resource_unavailable_error', 'code': 'resource_unavailable'}}"):
        super().__init__(message)
        self.message = message
        self.body = {"error": {"code": "resource_unavailable", "message": "Resource Unavailable"}}


class APITimeoutError(Exception):  # matched by class name, like the openai SDK class
    status_code = None


class FakeResponses:
    def __init__(self, errors, success=None):
        self.errors = list(errors)
        self.success = success
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return self.success


class FakeClient:
    def __init__(self, errors, success=None):
        self.responses = FakeResponses(errors, success)


class FlexRetryTest(unittest.TestCase):
    def setUp(self):
        self.sleeps = []
        sleep_patch = patch.object(fdi_classifier.time, "sleep", side_effect=self.sleeps.append)
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def test_flex_unavailable_retries_schedule_then_raises_typed_error(self):
        client = FakeClient([FakeRateLimitError() for _ in range(10)])
        with patch.object(fdi_classifier, "client", client):
            with self.assertRaises(fdi_classifier.FlexUnavailableError) as raised:
                fdi_classifier.create_flex_response(model="gpt-5.2", input=[])
        self.assertEqual(list(fdi_classifier.FLEX_RETRY_BACKOFF_SECONDS), self.sleeps)
        self.assertEqual(len(fdi_classifier.FLEX_RETRY_BACKOFF_SECONDS) + 1, len(client.responses.calls))
        self.assertTrue(all(call["service_tier"] == "flex" for call in client.responses.calls))
        self.assertIsInstance(raised.exception, RuntimeError)
        self.assertIn("Resource Unavailable", str(raised.exception))
        self.assertIsInstance(raised.exception.last_error, FakeRateLimitError)

    def test_timeout_is_treated_as_flex_unavailable(self):
        client = FakeClient([APITimeoutError("Request timed out.") for _ in range(10)])
        with patch.object(fdi_classifier, "client", client):
            with self.assertRaises(fdi_classifier.FlexUnavailableError):
                fdi_classifier.create_flex_response(model="gpt-5.2", input=[])
        self.assertEqual([20, 60, 120], self.sleeps)

    def test_transient_flex_error_then_success_keeps_flex_tier(self):
        client = FakeClient([FakeRateLimitError()], success="ok")
        with patch.object(fdi_classifier, "client", client):
            self.assertEqual("ok", fdi_classifier.create_flex_response(model="gpt-5.2", input=[]))
        self.assertEqual([20], self.sleeps)
        self.assertEqual(["flex", "flex"], [c["service_tier"] for c in client.responses.calls])

    def test_other_errors_propagate_unchanged_without_retry(self):
        class FakeBadRequestError(Exception):
            status_code = 400

        client = FakeClient([FakeBadRequestError("invalid request")])
        with patch.object(fdi_classifier, "client", client):
            with self.assertRaises(FakeBadRequestError):
                fdi_classifier.create_flex_response(model="gpt-5.2", input=[])
        self.assertEqual([], self.sleeps)
        self.assertEqual(1, len(client.responses.calls))

    def test_plain_429_without_resource_unavailable_is_not_retried(self):
        class FakeRateLimitError(Exception):
            status_code = 429

        client = FakeClient([FakeRateLimitError("Error code: 429 - rate limit reached for gpt-5.2")])
        with patch.object(fdi_classifier, "client", client):
            with self.assertRaises(FakeRateLimitError):
                fdi_classifier.create_flex_response(model="gpt-5.2", input=[])
        self.assertEqual([], self.sleeps)

    def test_predict_fdi_from_images_surfaces_flex_unavailable(self):
        """End to end through the gpt-5.2 branch of request_once with hosted image URLs (no upload)."""
        rng = np.random.default_rng(0)
        vertices = rng.random((60, 3))
        faces = np.arange(60).reshape(20, 3)
        face_labels = np.repeat(np.arange(1, 5), 5)
        client = FakeClient([FakeRateLimitError() for _ in range(10)])
        with patch.object(fdi_classifier, "client", client), \
                patch.object(fdi_classifier, "init_llm_client"), \
                patch.object(fdi_classifier.time, "sleep", side_effect=self.sleeps.append):
            with self.assertRaises(fdi_classifier.FlexUnavailableError):
                fdi_classifier.predict_fdi_from_images(
                    vertices, faces, face_labels, renderer=None,
                    gpt_model=fdi_classifier.MODELS["chatgpt-5"],
                    images=["https://example.invalid/a.png", "https://example.invalid/b.png"],
                )
        self.assertEqual([20, 60, 120], self.sleeps)
        self.assertEqual(4, len(client.responses.calls))
        self.assertTrue(all(c["service_tier"] == "flex" for c in client.responses.calls))


if __name__ == "__main__":
    unittest.main()
