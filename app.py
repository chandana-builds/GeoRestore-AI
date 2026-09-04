import os
import sys
import runpy

root_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(root_dir, "app")
for p in [root_dir, app_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

target_script = os.path.join(app_dir, "app.py")
runpy.run_path(target_script, run_name="__main__")
