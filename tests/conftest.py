import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hypothesis import settings
settings.register_profile("veda", max_examples=300, deadline=None, derandomize=True)
settings.load_profile("veda")
