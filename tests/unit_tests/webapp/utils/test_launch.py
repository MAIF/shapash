"""Unit tests for ``run_in_background`` (serve a WSGI app on a killable background thread)."""

import unittest
import urllib.request

from shapash.webapp.utils.launch import RunningApp, run_in_background


def _wsgi_app(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"ok"]


class TestRunInBackground(unittest.TestCase):
    def test_serves_and_returns_killable_handle(self):
        app = run_in_background(_wsgi_app, "127.0.0.1", 0)
        try:
            self.assertIsInstance(app, RunningApp)
            self.assertTrue(app.is_alive())
            with urllib.request.urlopen(app.url, timeout=5) as resp:
                self.assertEqual(resp.read(), b"ok")
        finally:
            app.kill()
        self.assertFalse(app.is_alive())

    def test_context_manager_kills_on_exit(self):
        with run_in_background(_wsgi_app, "127.0.0.1", 0) as app:
            self.assertTrue(app.is_alive())
        self.assertFalse(app.is_alive())
