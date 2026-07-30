"""
Day 2 Homework Solution Suite — Master Runner Script
Executes all 5 homework solution scripts sequentially and outputs figures to day_2/hw/plots/
"""

import subprocess
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
scripts = [
    "01_fig3_errorbars.py",
    "02_fig4_shared_y.py",
    "03_fig5_shared_clim.py",
    "04_fig7_grid3x3.py",
    "05_fig8_composite.py"
]

python_exe = sys.executable

print("="*70)
print("EXECUTING DAY 2 HOMEWORK SOLUTION SUITE")
print("="*70)

for script in scripts:
    script_path = os.path.join(script_dir, script)
    print(f"\n---> Running {script}...")
    res = subprocess.run([python_exe, script_path], capture_output=True, text=True)
    if res.returncode == 0:
        print(res.stdout.strip())
    else:
        print(f"Error running {script}:")
        print(res.stderr)
        sys.exit(res.returncode)

print("\n" + "="*70)
print("ALL 5 HOMEWORK SOLUTION SCRIPTS EXECUTED SUCCESSFULLY!")
print("="*70)
