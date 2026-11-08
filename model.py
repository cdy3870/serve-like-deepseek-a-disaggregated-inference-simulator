"""
Serve Like DeepSeek: A Disaggregated Inference Simulator

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_workload
import math
import numpy as np

CLASS_SPEC = {"chat": (2000, 0.6, 16000, 300, 0.3), "agent": (40000, 0.4, 64000, 150, 0.8), "doc": (150000, 0.5, 400000, 500, 0.1)}


def make_workload(n, seed, rate=4.0, mix=(0.7, 0.2, 0.1)):
    # TODO: gaps, classes, then per-request ctx / cached / out in that order
    
    rng = np.random.default_rng(seed)

    gaps = rng.exponential(1 / rate, size=n)
    t_arrive = np.cumsum(gaps)

    classes = rng.choice(["chat", "agent", "doc"], size=n, p=list(mix))

    workload = []

    for i in range(n):
        cls = classes[i]
        ctx = int(min(CLASS_SPEC[cls][2], max(128, rng.lognormal(CLASS_SPEC[cls][0], CLASS_SPEC[cls][1]))))
        cached = int(ctx * CLASS_SPEC[cls][-1] * rng.uniform(0.5, 1.0)) // 128 * 128
        out = int(max(1, rng.poisson(CLASS_SPEC[cls][-2])))

        workload.append({"id":i, "t_arrive": t_arrive[i], "cls": str(cls), "ctx": ctx, "cached": cached, "out": out})

    return workload

