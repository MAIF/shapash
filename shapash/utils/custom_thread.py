"""
Override threading custom module
"""

import sys
import threading
from collections.abc import Callable
from types import FrameType
from typing import Any

TraceCallback = Callable[[FrameType, str, Any], Any]


class CustomThread(threading.Thread):
    """
    Thread subclass that can be stopped from another Python object.

    Stopping is cooperative: the traced thread raises ``SystemExit`` at its
    next line event after ``kill`` is called.

    Parameters
    ----------
    *args : Any
        Positional arguments forwarded to :class:`threading.Thread`.
    on_kill : Callable[[], None], optional
        Callback invoked synchronously in the calling thread before the
        traced thread is marked for stopping.
    **keywords : Any
        Keyword arguments forwarded to :class:`threading.Thread`.
    """

    def __init__(
        self,
        *args: Any,
        on_kill: Callable[[], None] | None = None,
        **keywords: Any,
    ) -> None:
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False
        self.__run_backup: Callable[[], None] = self.run
        self.on_kill = on_kill

    def start(self) -> None:
        """Starts the thread"""
        self.__run_backup = self.run
        object.__setattr__(self, "run", self.__run)
        threading.Thread.start(self)

    def __run(self) -> None:
        sys.settrace(self.globaltrace)
        self.__run_backup()
        object.__setattr__(self, "run", self.__run_backup)

    def globaltrace(self, frame: FrameType, event: str, arg: Any) -> TraceCallback | None:
        """
        Track the global trace
        """
        if event == "call":
            return self.localtrace
        else:
            return None

    def localtrace(self, frame: FrameType, event: str, arg: Any) -> TraceCallback:
        """
        Track the local trace
        """
        if self.killed:
            if event == "line":
                raise SystemExit()
        return self.localtrace

    def kill(self) -> None:
        """
        Kill the current Thread
        """
        if self.on_kill is not None:
            self.on_kill()
        self.killed = True
