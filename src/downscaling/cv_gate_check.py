"""
cv_gate_check.py

Implements Task 4 + the gate for Task 5 of Phase 2.6.

PRE-REGISTERED THRESHOLD (fixed here, before any CV result is seen):
    CV_IMPROVEMENT_THRESHOLD = 0.02

This number is fixed at the time this script is written, not adjusted
after seeing cv_tune.py's output. If the new-feature CV result comes in
at, say, +0.015, that is a documented "no" — this script will print that
verdict and refuse to recommend a held-out run. Changing the threshold
after the fact defeats the point of pre-registering it; if you genuinely
want to revisit the threshold, do it in a separate, dated note in
validation_methodology.md, not by editing this constant retroactively.

WHAT THIS DOES
--------------
1. Runs (or loads results from) cv_tune.py TWICE:
     - baseline: current feature set (Section 9), current best config
     - candidate: feature set + 23 soil/LULC one-hot channels, re-tuned
       from scratch (not just re-scored with the old hyperparameters —
       see feature_utils_patch.py's note on why re-tuning matters here)
2. Compares mean spatially-blocked CV R² (Section 11's methodology —
   8x8 spatial block folds, NOT date-grouped folds).
3. Applies the pre-registered threshold and prints an explicit verdict.

ASSUMPTIONS (verify against your actual cv_tune.py):
  - cv_tune.py can be invoked programmatically and returns a dict/object
    exposing `.mean_r2` and `.std_r2` for a given feature-set config. If
    your actual cv_tune.py is CLI-only, wrap it with subprocess + parse
    its printed output instead of the direct import below.
"""

CV_IMPROVEMENT_THRESHOLD = 0.02  # pre-registered — see module docstring


def run_cv(feature_set_name: str):
    """
    Stub for invoking your actual cv_tune.py.
    Replace this with either:
      (a) a direct import + call into cv_tune's tuning function, or
      (b) subprocess.run(["python", "cv_tune.py", "--features", feature_set_name])
          followed by parsing its printed mean/std R².
    """
    raise NotImplementedError(
        "Wire this to your actual cv_tune.py invocation before running. "
        "This script only encodes the comparison logic and the "
        "pre-registered gate, not the CV run itself."
    )


def main():
    baseline = run_cv("baseline_section9")
    candidate = run_cv("baseline_section9_plus_soil_lulc")

    delta = candidate.mean_r2 - baseline.mean_r2

    print("=== Phase 2.6 Training-Only Spatially-Blocked CV Comparison ===")
    print(f"Baseline  (Section 9 features): mean R² = {baseline.mean_r2:.3f} "
          f"(std {baseline.std_r2:.3f})")
    print(f"Candidate (+ soil/LULC):        mean R² = {candidate.mean_r2:.3f} "
          f"(std {candidate.std_r2:.3f})")
    print(f"Delta: {delta:+.3f}")
    print(f"Pre-registered threshold: +{CV_IMPROVEMENT_THRESHOLD:.3f}")
    print()

    if delta > CV_IMPROVEMENT_THRESHOLD:
        print("VERDICT: PASS. Candidate clears the pre-registered threshold.")
        print("-> Proceed to the single, final held-out check "
              "(validate_downscale.py), retraining the final RF on the "
              "full training-well set with the soil/LULC features included.")
        print("-> This is the last permitted touch of the held-out set for "
              "this project. Accept and report whatever number results, "
              "per Section 12's discipline.")
    else:
        print("VERDICT: NO GO. Candidate does not clear the pre-registered "
              "threshold.")
        print("-> Do NOT run validate_downscale.py against the held-out set "
              "for this feature addition. Document this as a third negative "
              "result alongside the Phase 2.5 deficit-feature and "
              "residual-Kriging experiments, and close out Phase 2.6 without "
              "a held-out touch.")


if __name__ == "__main__":
    main()