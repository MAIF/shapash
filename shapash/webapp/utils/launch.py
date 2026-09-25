"""Stop handle for a Dash app served in the background by Dash's own Jupyter integration.

``dash.Dash.run`` already picks the right serving strategy: in a script it blocks on Flask's server
(clean Ctrl+C, request logs); in Jupyter it serves from a background thread and returns at once. The
one thing it doesn't give a notebook user is a way to stop that background server, so this wraps the
server Dash registered for ``(host, port)``. Only ``jupyter_dash._servers`` is private here — Dash's
own port-reuse logic relies on it, and ``tests/unit_tests/webapp/utils/test_launch.py`` pins it.
"""

from __future__ import annotations

from dash import jupyter_dash


class RunningApp:
    """A Dash app Dash is serving in the background of a notebook; call ``.kill()`` to stop it."""

    def __init__(self, host: str, port: int):
        self.url = f"http://{host}:{port}/"
        self._key = (host, port)
        self._server = jupyter_dash._servers[self._key]

    def kill(self) -> None:
        """Stop the server; a no-op if it is already stopped."""
        self._server.shutdown()
        # Re-running an app on the same port replaces the registry entry — never evict a newer server.
        if jupyter_dash._servers.get(self._key) is self._server:
            del jupyter_dash._servers[self._key]

    def is_alive(self) -> bool:
        """Whether this server is still the one Dash is serving on its port."""
        return jupyter_dash._servers.get(self._key) is self._server

    def __enter__(self) -> RunningApp:
        return self

    def __exit__(self, *exc_info) -> None:
        self.kill()

    def __repr__(self) -> str:
        return f"<RunningApp url={self.url!r} running={self.is_alive()}>"
