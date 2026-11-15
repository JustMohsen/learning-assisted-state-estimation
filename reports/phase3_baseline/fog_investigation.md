# FOG Drift Investigation (Phase 3 revision, R5)

*15 November 2026. Question from the Phase 3 review: why does FOG-integrated heading drift by tens of degrees per hour against the reference, and is it the FOG, the reference, or our model?*

## Method

1. **FOG vs graph-node rotations** (LiDAR/RTK-anchored).
   - For each pair of consecutive graph nodes (~3.6 s apart), compute the reference's 3D body-frame rotation about body z: the z component of `Log(C_k^T C_k+1)`.
   - Compare it with the sign-corrected FOG increment.
   - Per session, fit residual = bias·dt + scale·rotation.
2. **FOG vs GPS course.**
   - On straight, fast stretches (< 3° turn over 20 s, > 20 m travelled), GPS course equals heading.
   - Compare FOG-integrated heading drift against GPS course (no FOG, no LiDAR involved) and against the reference.
3. **Structure:** regress the node-interval residual rate on scale, roll rate, pitch rate, |pitch rate| and time since start (within-session); compare early vs late bias and winter vs summer.
4. **FOG axis vs IMU** (no reference): regress FOG increments on the three IMU gyro increments.
5. **Datasheet** check.

## Results

| Check | Result |
|---|---|
| Per-session bias vs nodes | **−25 … +28°/h**, varying within sessions (first vs last third often differ by 20–40°/h); scale about −0.1 to −0.2% |
| GPS course vs reference | FOG drifts at about the same rate against both (−9 … +16°/h session trend): **the drift is in the FOG**, not the reference |
| Warm-up (time since start) | No effect (0.1 ± 0.3°/h per minute); early-vs-late differences change sign randomly |
| Season | Winter 7.1°/h vs summer 5.0°/h mean (large spread): no clear seasonal effect |
| Pitch rate / \|pitch rate\| | Negligible |
| Roll rate | Strong correlation with the *reference* residual (coefficient ≈ 0.17), but the FOG axis is within **0.7°** of the IMU z axis with unit scale (check 4). So this is a limitation of the **reference's roll at the graph nodes**, not of the sensors. |
| KVH DSP-3000 datasheet (original, digital) | **Bias offset ±20°/h** (turn-on), **bias vs temperature ≤ 6°/h** (1σ), bias instability ≤ 1°/h at constant 25 °C, ARW ≤ 0.067°/√h ([KVH datasheet](https://www.canalgeomatics.com/wp-content/uploads/2019/11/kvh-dsp-3000-fog-datasheet.pdf); the later EMCORE spec gives ±5°/h and 3°/h, [EMCORE](https://emcore.com/products/dsp-3000-fiber-optic-gyro-fog-non-itar/attachment/dsp-3000-fiber-optic-gyro/)) |

## Conclusions

1. **The drift is a genuine FOG bias:** a per-session turn-on offset of up to about ±25°/h, plus slow within-session (temperature) variation. This matches the original datasheet. My filter prior (1e-5 rad/s ≈ 2°/h, from the constant-temperature *instability* figure) was about 10–20× too small, so the filter could not estimate it.
2. **Proposed FOG bias model:**
   - prior σ = 20°/h ≈ **1e-4 rad/s** (turn-on offset);
   - random walk ≈ **1e-6 to 2e-6 rad/s/√s** (allows ~5–20°/h change over 30–60 min).

   To be confirmed in the R7 tuning, together with the GPS error states. Without them, a wider prior made results worse, because correlated GPS error was being absorbed into the bias.
3. **FOG axis:** aligned with the IMU z axis to 0.7°, with no scale error. No correction needed (a 0.7° tilt leaks about 1% of pitch/roll rate).
4. **Earth rate: compensated (user decision, Phase 3 review).**
   - The FOG and gyros measure inertial rotation: about 10.1°/h on the local vertical at 42.3° N. A filter navigating in local North-East-Down must remove it.
   - My earlier test, where compensation made heading worse, was confounded by the too-tight FOG-bias prior (1e-5 rad/s), so the filter could not absorb the per-session offset (+6 ± 3°/h mean residual; per-session offsets up to ±25°/h).
   - The realistic bias prior now leaves the offsets to the bias states (and later the network). R7 includes an "off" control.
5. **Reference roll at graph nodes** does not follow the IMU's roll motion closely. Short-window attitude metrics against the reference should be read with that in mind.
