"""Serve a WSGI app in a background thread, stoppable from the caller.

Mirrors how Dash's own Jupyter integration launches apps (``dash._jupyter.JupyterDash.run_app``):
a daemon thread running ``server.serve_forever()``, stopped via ``server.shutdown()`` — the
cooperative shutdown built into ``socketserver.BaseServer``. No thread-killing trickery is needed
since ``serve_forever``/``shutdown`` are already a clean, designed-for-this pair.
"""

from __future__ import annotations

import threading

from werkzeug.serving import BaseWSGIServer, make_server


class RunningApp:
    """A WSGI app running on a background thread; call ``.kill()`` to stop it."""

    def __init__(self, server: BaseWSGIServer, thread: threading.Thread, url: str):
        self._server = server
        self._thread = thread
        self.url = url

    def kill(self) -> None:
        """Stop the server and wait for its thread to exit."""
        self._server.shutdown()
        self._thread.join()

    def is_alive(self) -> bool:
        """Whether the server thread is still running."""
        return self._thread.is_alive()

    def __enter__(self) -> RunningApp:
        return self

    def __exit__(self, *exc_info) -> None:
        self.kill()

    def __repr__(self) -> str:
        return f"<RunningApp url={self.url!r} running={self.is_alive()}>"


def run_in_background(wsgi_app, host: str, port: int) -> RunningApp:
    """Serve ``wsgi_app`` on a background daemon thread and return a killable handle.

    Parameters
    ----------
    wsgi_app : WSGI application
        The application to serve (e.g. a Flask app, or ``dash.Dash(...).server``).
    host : str
        Host to bind the server to.
    port : int
        Port to bind the server to.

    Returns
    -------
    RunningApp
        Already-started server handle; call ``.kill()`` on it (or use it as a context manager)
        to stop the server.
    """
    server = make_server(host, port, wsgi_app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return RunningApp(server, thread, url=f"http://{host}:{server.server_port}/")
