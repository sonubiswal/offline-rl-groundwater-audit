Trishna-OPAL --- Ablation Experiment Results
Structured Experimental Results
Test ID Experiment Hypothesis Feature / Method Added Baseline Test R² ΔR² Baseline Test ΔRMSE (m) Baseline Test ΔMAE (m) N Verdict
R² RMSE (m) RMSE MAE (m) MAE (m)
(m)

E01 Cumulative Groundwater levels chirps_cum12_anomaly 0.198 0.190 −0.008 5.226 5.246 +0.020 3.371 3.360 −0.011 4,753 No
Rainfall respond to --- 12-month cumulative improvement
Deficit accumulated rainfall rainfall anomaly (flat)
deficits over
multiple months
(drought memory),
which a single-month
rainfall feature
cannot capture.
Adding a 12-month
cumulative anomaly
should improve
predictions during
dry spells.

E01 --- Cumulative Rainfall Deficit
Hypothesis
Groundwater levels may exhibit multi-month drought memory, meaning a
12-month cumulative rainfall anomaly could provide information that a
single-month rainfall variable cannot capture.

Method
Engineered chirps_cum12_anomaly.

Calculated it from the 12-month cumulative rainfall anomaly relative
to monthly climatology.

Added the feature to the existing Random Forest feature set.

Retrained the model using the same spatial hold-out strategy.

Results
R²: 0.198 → 0.190 (ΔR² = −0.008)

RMSE: 5.226 → 5.246 m (ΔRMSE = +0.020 m)

MAE: 3.371 → 3.360 m (ΔMAE = −0.011 m)

Feature importance: 0.052

Conclusion
The feature was used by the Random Forest during training, but the
additional information did not improve generalization to held-out
wells.

Interpretation
Large-scale cumulative rainfall deficit alone does not adequately
explain the remaining prediction error. The residual error may instead
be associated with local hydrogeological variability, well-specific
conditions, aquifer properties, or other spatially heterogeneous
processes that are not sufficiently represented by cumulative
rainfall.

E02 --- Residual Kriging / Spatial Bias Correction
Hypothesis
The Random Forest may produce spatially correlated residuals. If so,
interpolating training-well residuals using Ordinary Kriging could
correct systematic spatial bias in held-out predictions.

Method
Generated RF predictions at training wells.

Calculated prediction residuals.

Applied Ordinary Kriging to interpolate the residual field.

Added the interpolated residual correction to held-out RF
predictions.

Evaluated using 10×10 spatial-block cross-validation to
reproduce the geometry of the true spatial hold-out.

Results
R²: 0.189 → 0.182 (ΔR² = −0.007)

RMSE: 5.230 → 5.252 m (ΔRMSE = +0.022 m)

MAE: 3.358 → 3.323 m (ΔMAE = −0.035 m)

Diagnostic spatial CV gain: +0.006 R²

Important Validation Finding
Random-fold CV initially suggested a much larger +0.041 R²
improvement, but this apparent gain largely disappeared when evaluated
using spatially blocked CV.

Conclusion
Residual Kriging did not provide a meaningful improvement under the
realistic spatial validation scheme. The residual field does not
appear to contain sufficient spatial structure at the distances
separating the held-out wells.

Interpretation
The apparent benefit under random-fold validation was likely influenced
by spatial leakage or overly optimistic validation geometry. The
spatially blocked experiment provides stronger evidence that residual
interpolation does not generalize reliably to genuinely unseen
locations.

Overall Scientific Findings
Finding E01: Rainfall Memory E02: Residual Kriging

Feature/method Yes Yes
utilized?

R² improvement? ❌ No ❌ No

RMSE improvement? ❌ No ❌ No

MAE improvement? ✅ Very small ✅ Small

Generalization ❌ No ❌ No
improvement?

Evidence of useful Weak Weak under spatial
additional information? validation

Main lesson 12-month rainfall Residual spatial
memory is insufficient structure is
insufficient

Key Takeaway
Both ablation experiments failed to improve the model under strict
spatial generalization, despite providing small reductions in MAE. E01
indicates that cumulative rainfall deficit alone does not capture the
remaining groundwater variability, while E02 demonstrates that
apparent gains from residual Kriging can largely disappear under
realistic spatially blocked validation. Together, these experiments
strengthen the conclusion that the remaining prediction error is
dominated by local, heterogeneous hydrogeological processes rather
than by simple temporal rainfall accumulation or smoothly varying
spatial bias.

