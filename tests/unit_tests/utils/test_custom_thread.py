import time

import pytest

from shapash.utils.custom_thread import CustomThread


class TestCustomThread:
    def test_init_defaults(self):
        t = CustomThread(target=lambda: None)
        assert t.killed is False
        assert t.on_kill is None

    def test_init_with_on_kill(self):
        t = CustomThread(target=lambda: None, on_kill=lambda: None)
        assert t.on_kill is not None

    def test_globaltrace_call_event_returns_localtrace(self):
        t = CustomThread(target=lambda: None)
        assert t.globaltrace(None, "call", None) == t.localtrace

    def test_globaltrace_other_event_returns_none(self):
        t = CustomThread(target=lambda: None)
        assert t.globaltrace(None, "line", None) is None

    def test_localtrace_not_killed_returns_localtrace(self):
        t = CustomThread(target=lambda: None)
        assert t.localtrace(None, "line", None) == t.localtrace

    def test_localtrace_killed_raises_systemexit_on_line_event(self):
        t = CustomThread(target=lambda: None)
        t.killed = True
        with pytest.raises(SystemExit):
            t.localtrace(None, "line", None)

    def test_localtrace_killed_non_line_event_returns_localtrace(self):
        t = CustomThread(target=lambda: None)
        t.killed = True
        assert t.localtrace(None, "call", None) == t.localtrace

    def test_kill_without_on_kill_callback(self):
        t = CustomThread(target=lambda: None)
        t.kill()
        assert t.killed is True

    def test_kill_calls_on_kill_callback(self):
        called = []
        t = CustomThread(target=lambda: None, on_kill=lambda: called.append(True))
        t.kill()
        assert called == [True]
        assert t.killed is True

    def test_start_and_kill_running_thread(self):
        def loop():
            while True:
                time.sleep(0.01)

        t = CustomThread(target=loop)
        t.start()
        time.sleep(0.05)
        assert t.is_alive()
        t.kill()
        t.join(timeout=2)
        assert not t.is_alive()
