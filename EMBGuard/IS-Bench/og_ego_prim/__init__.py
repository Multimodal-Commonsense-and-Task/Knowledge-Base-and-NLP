"""
og_ego_prim package initialization
Adds project root to sys.path to enable importing src modules
"""
import os
import sys

# Add project root to path to enable importing src modules as a package
# Project root is 2 levels up from og_ego_prim (IS-Bench/og_ego_prim -> IS-Bench -> EMBGuard)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
