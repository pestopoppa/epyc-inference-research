"""Declared production/measurement identities shared by campaign and live controls.

These are the existing campaign values, with no deployment or authority changes.
"""
import os

POST_T0_QUIET_BARRIER_S = 65.0

POST_T0_QUIET_SAMPLES = 3

POST_T0_QUIET_SAMPLE_INTERVAL_S = 5.0

IQK_MATCHED_PAIR_REPS = 1

PRODUCTION_REPO = "/mnt/raid0/llm/llama.cpp"

PRODUCTION_BRANCH = "production-consolidated-v10"

PRODUCTION_COMMIT = "ffc1bac82eeca6f9099e1ccd9ba49703c460a115"

MEASUREMENT_REPO = PRODUCTION_REPO

MEASUREMENT_BRANCH = PRODUCTION_BRANCH

MEASUREMENT_COMMIT = PRODUCTION_COMMIT

MEASUREMENT_BUILD_ROOT = os.path.join(
    "/mnt/raid0/llm/tmp", f"build-ak-t0-cpu-{PRODUCTION_COMMIT[:9]}")
