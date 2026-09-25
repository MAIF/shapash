"""Unit tests for ``RunningApp`` — the stop handle over a server Dash's Jupyter integration registered."""

import threading
import unittest
import urllib.error
import urllib.request

from dash import jupyter_dash
from werkzeug.serving import make_server

from shapash.webapp.utils.launch import RunningApp


def _wsgi_app(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"ok"]


def _serve_like_jupyter_dash(host="127.0.0.1"):
    """Start and register a server the way ``dash._jupyter.JupyterDash.run_app`` does."""
    server = make_server(host, 0, _wsgi_app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port
    jupyter_dash._servers[(host, port)] = server
    return host, port, server


class TestRunningApp(unittest.TestCase):
    def test_dash_still_exposes_server_registry(self):
        # RunningApp's only dependency on Dash internals — fail loudly if an upgrade renames it.
        self.assertIsInstance(jupyter_dash._servers, dict)

    def test_kill_stops_server_and_unregisters_it(self):
        host, port, _ = _serve_like_jupyter_dash()
        app = RunningApp(host, port)
        with urllib.request.urlopen(app.url, timeout=5) as resp:
            self.assertEqual(resp.read(), b"ok")
        self.assertTrue(app.is_alive())

        app.kill()

        self.assertFalse(app.is_alive())
        self.assertNotIn((host, port), jupyter_dash._servers)
        with self.assertRaises(urllib.error.URLError):
            urllib.request.urlopen(app.url, timeout=1)

    def test_kill_twice_is_a_no_op(self):
        app = RunningApp(*_serve_like_jupyter_dash()[:2])
        app.kill()
        app.kill()
        self.assertFalse(app.is_alive())

    def test_context_manager_kills_on_exit(self):
        with RunningApp(*_serve_like_jupyter_dash()[:2]) as app:
            self.assertTrue(app.is_alive())
        self.assertFalse(app.is_alive())

    def test_kill_does_not_evict_a_newer_server_on_the_same_port(self):
        host, port, _ = _serve_like_jupyter_dash()
        old = RunningApp(host, port)
        newer = make_server(host, 0, _wsgi_app)
        jupyter_dash._servers[(host, port)] = newer  # what re-running the app on that port does
        try:
            old.kill()
            self.assertIs(jupyter_dash._servers[(host, port)], newer)
        finally:
            del jupyter_dash._servers[(host, port)]
            newer.server_close()
