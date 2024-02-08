"""
Override threading custom module
"""
import sys
import threading


class CustomThread(threading.Thread):
    """
    Python ovveride threading class
    Used to kill a thread from python object
    Parameters
    ----------
    threading : threading.Thread
        Thread which you want to instanciate
    """

    def __init__(self, *args, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False
        self.__run_backup = None

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
        self.killed = True
