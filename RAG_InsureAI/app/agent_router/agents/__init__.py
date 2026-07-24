"""
Importing this package registers every known agent (import side-effects
call register_agent() — see registry.py). Adding a new agent later means
one new file here + one import line below; nothing else changes.
"""
from . import ava  # noqa: F401
