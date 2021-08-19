import unittest
from shapash.explainer.smart_explainer import SmartExplainer
from pathlib import Path
from os import path
import sys


class TestWebappSettings(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        current = Path(path.abspath(__file__)).parent.parent.parent
        self.xpl = SmartExplainer()
        if str(sys.version)[0:3] == '3.6':
            pkl_file = path.join(current, 'data/xpl_to_load_36.pkl')
        elif str(sys.version)[0:3] == '3.7':
            pkl_file = path.join(current, 'data/xpl_to_load_37.pkl')
        elif str(sys.version)[0:3] == '3.8':
            pkl_file = path.join(current, 'data/xpl_to_load_38.pkl')
        elif str(sys.version)[0:3] == '3.9':
            pkl_file = path.join(current, 'data/xpl_to_load_39.pkl')
        else:
            raise NotImplementedError
        self.xpl.load(pkl_file)
        super(TestWebappSettings, self).__init__(*args, **kwargs)

    def test_settings_types(self):
        settings = {'rows': None,
                    'points': 5200.4,
                    'violin': -1,
                    'features': "oui"}
        self.xpl.init_app(settings)
        print(self.xpl.smartapp.settings)
        assert all(isinstance(attrib, int) for k, attrib in self.xpl.smartapp.settings.items())

    def test_settings_values(self):
        settings = {'rows': 0,
                    'points': 5200.4,
                    'violin': -1,
                    'features': "oui"}
        self.xpl.init_app(settings)
        assert all(attrib > 0 for k, attrib in self.xpl.smartapp.settings.items())

    def test_settings_keys1(self):
        settings = {}
        self.xpl.init_app(settings)
        assert all(k in ['rows', 'points', 'violin', 'features'] for k in self.xpl.smartapp.settings)

    def test_settings_keys2(self):
        settings = {'oui': 1,
                    1: 2,
                    "a": []}
        self.xpl.init_app(settings)
        assert all(k in ['rows', 'points', 'violin', 'features'] for k in self.xpl.smartapp.settings)
