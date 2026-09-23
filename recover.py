"""Bring the arm home from wherever it is: swing joint 1 to the requested azimuth (away from any object),
lift joints 2/3 to a folded pose, then glide to REST.   python recover.py [--azimuth 0.0] [--dry]"""
import sys, time, numpy as np
import yam_mac  # noqa: F401
from pick_bottle import Arm, Planner, REST, goto_joint

az = float(sys.argv[sys.argv.index("--azimuth") + 1]) if "--azimuth" in sys.argv else 0.0
dry = "--dry" in sys.argv
planner = Planner()
arm = Arm(False, "can0", 100.0)
try:
    q = arm.q(); print("current q:", np.round(q, 2), "fk:", np.round(planner.fk_pos(q), 3))
    swing = q.copy(); swing[0] = az
    lifted = np.array([az, 0.7, 1.1, -1.3, 0.0, 0.0])
    for name, qt in (("swing", swing), ("lift", lifted), ("rest", REST)):
        print(f"{name}: q {np.round(qt, 2)} fk {np.round(planner.fk_pos(qt), 3)} safe {planner.path_is_safe(arm.q() if not dry else q, qt)}")
        if not dry:
            goto_joint(arm, planner, qt, 4.0); time.sleep(0.3)
        else:
            q = qt
finally:
    arm.close()
