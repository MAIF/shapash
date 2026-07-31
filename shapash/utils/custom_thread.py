"""
Override threading custom module
"""

import sys
import threading
from collections.abc import Callable


class CustomThread(threading.Thread):
    """
    Python ovveride threading class
    Used to kill a thread from python object
    Parameters
    ----------
    threading : threading.Thread
        Thread which you want to instanciate
    on_kill : Callable, optional
        Extra callback invoked when the thread is killed, in addition to
        stopping the traced run loop (e.g. to shut down a server bound to
        this thread).
    """

    def __init__(self, *args, on_kill: Callable[[], None] | None = None, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False
        self.__run_backup = None
        self.on_kill = on_kill

    def start(self):
        """Starts the thread"""
        self.__run_backup = self.run
        self.run = self.__run
        threading.Thread.start(self)

    def __run(self):
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def globaltrace(self, frame, event, arg):
        """
        Track the global trace
        """
        if event == "call":
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, event, arg):
        """
        Track the local trace
        """
        if self.killed:
            if event == "line":
                raise SystemExit()
        return self.localtrace

    def kill(self):
        """
        Kill the current Thread
        """
        if self.on_kill is not None:
            self.on_kill()
        self.killed = True
