# NCLT: sensor and state specification

Updated: 14 November 2026. NCLT, five planar outputs, tilt-aware internal attitude and **error-state EKF/MSCKF with bounded history and no initial persistent map/loop closure are locked**. IMU/FOG propagation, wheel/GNSS/visual updates and three learning roles are agreed. Use frozen SuperPoint/LightGlue and instance masks, sensor exclusions, camera configurability and removal diagnostics. Exact fusion, state augmentation, parameter lifecycle and neural interfaces remain open. Equations use plain text.

## Evidence inspected

This pass checked the official dataset documentation, its sample readers, the papers/code listed below, one complete public **2013-01-10 sensor archive** (21,669,014 compressed bytes), and the small corrected camera-calibration ZIP. The sensor archive's included README documents the FOG/wheel additions made after the original paper. No image/LiDAR archive was downloaded, no full dataset was staged, and no estimator was run. One recording's audit does not establish completeness or rates for every recording.

- [Official NCLT site and download links](https://robots.engin.umich.edu/nclt/).
- [Dataset paper, particularly Sections 2, 4, 6 and 7](https://robots.engin.umich.edu/nclt/nclt.pdf).
- [Inspected sensor archive, including README.txt](https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/sensor_data/2013-01-10_sen.tar.gz).
- [Official GPS reader](https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/python/read_gps.py) and [IMU reader](https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/python/read_ms25.py).

## Available inputs and their meaning

| Sensor/product | Released information | Proposed role / important processing |
|---|---|---|
| Microstrain IMU, ms25.csv | Timestamp; three magnetometer, accelerometer and gyro axes | Retain released sensor-frame vectors. Correct calibration and propagate their noise inside the joint estimator before attitude rotation/gravity compensation. Internally filtered sensor outputs, not ADC-level data |
| IMU attitude, ms25_euler.csv | Timestamp; roll, pitch, heading in radians | Optional reproduction/diagnostic or declared initialization product. Do not freeze it as the main estimator's orientation and assume independent noise |
| Wheels, wheels.csv | Timestamp; left and right wheel speeds in m/s | Forward-speed/kinematic observations. Already converted to speed; not raw encoder ticks. Handle timing, scale and slip |
| KVH DSP-3000 FOG, kvh.csv | Timestamp; heading in radians, per archive README | Additional rotation information. Unwrap angles and form increments with measured time intervals; account for mounting and drift |
| Consumer GPS, gps.csv | Eight columns: timestamp, fix mode, satellite count, latitude, longitude, altitude, track, speed | Use receiver position fixes as observations after coordinate conversion, antenna lever-arm compensation and quality/outlier checks |
| Ladybug3 camera system | Six views: five around the horizontal ring and one upward, recorded at 5 Hz | Frozen SuperPoint + LightGlue followed by geometric processing; select initial view(s) after checking calibration/coverage. This is not a conventional rectified stereo pair |
| NovAtel RTK GPS, gps_rtk.csv | Separate higher-grade receiver's solution; corresponding RTK/reference error file also released | Proposed reference/diagnostic role, excluded from estimator inputs in the main experiment |
| Velodyne 3D and two Hokuyo planar LiDARs | Additional perception streams | Available, but adding LiDAR changes the sensor budget, frontend and reference dependence; not proposed for the first main estimator |
| Barometer | No pressure stream in the documented release or inspected archive | Do not design a pressure update around this dataset |

FOG means **fiber-optic gyroscope**. It is a separate, more stable rotation sensor, not a source of absolute north-referenced heading merely because its file column is called heading. Its integrated angle has an arbitrary reference and can drift. The [manufacturer manual](https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/manuals/DSP3000_Manual.pdf) distinguishes angular rate, incremental angle and integrated angle; NCLT's archive README establishes that the released column is heading in radians. Do not integrate that column again as though it were angular rate.

Magnetic data can be part of an attitude filter. An accelerometer/gyro IMU alone has no magnetometer; a magnetic AHRS uses one as an additional observation. The [Microstrain theory manual](https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/manuals/3DM-GX3-45_Theory_of_Operation.pdf) describes magnetic heading correction and its optional disabling under interference. We have not established the exact logged device configuration. If we consume vendor heading, it may already contain magnetic information; that matters when claiming a sensor subset. **User decision: disable magnetic updates initially; they may remain excluded indefinitely.** No future magnetic experiment is required. Disturbance and systematic distortion, rather than white-noise magnitude alone, are the concern.

### GNSS: receiver fixes, not satellite-level measurements

The documented release and inspected archive do not supply raw pseudoranges, carrier phase, satellite Doppler, satellite IDs/ephemerides, per-satellite elevation/azimuth or C/N0. A satellite-count column is aggregate metadata, not individual satellite observations. NCLT therefore supports receiver-solution-level GNSS fusion, commonly called loosely coupled GNSS/inertial fusion; it does not directly support rebuilding a raw-observation GNSS solver or learning per-satellite weights.

The published GPS format also lacks a full per-fix position covariance and DOP fields. Track and speed come from the same receiver and are not independent sensors. The paper's track-unit label is ambiguous; validate its semantics/units before using it. Initial proposal: fuse horizontal positions first; consider speed/course only after their interpretation and noise relationship are verified.

**Concrete audit finding:** all 7,186 satellite-count entries in this recording's consumer GPS file are zero, despite fix-mode entries of 2 or 3. A zero satellite count cannot automatically be treated as invalid here. A quality feature must be checked across the selected recordings before use. Modes 2 and 3 document valid horizontal position; mode 3 additionally indicates valid altitude. Reproducing another implementation's mode-3-only gate is a separate protocol choice.

Other checks on this recording:

| Stream | Rows | Mean logged row rate over timestamp span |
|---|---:|---:|
| ms25.csv | 48,324 | 47.099 Hz |
| ms25_euler.csv | 48,315 | 47.091 Hz |
| wheels.csv | 42,276 | 41.206 Hz |
| kvh.csv | 97,007 | 94.550 Hz |
| gps.csv | 7,186 | 7.005 Hz |
| gps_rtk.csv | 2,791 | 2.722 Hz |

These are **CSV row rates**, not independently verified hardware fix/update rates. Timestamps are irregular. There are 2,582 consecutive unchanged consumer-GPS latitude/longitude/altitude triples; unchanged positions can also occur during stops. Do not equate every row with a new independent position observation, nor blindly remove all repeated positions. The paper's nominal hardware rates alone are insufficient for our timing/noise implementation. Preserve actual timestamps and audit duplicates, held values and gaps.

### What processing is needed apart from vision?

1. Parse timestamps in microseconds, check monotonicity/gaps and use actual time intervals. Keep causal updates at each stream's available cadence; repeated use of a held sample does not create independent information.
2. Transform sensor measurements using the published mounting geometry. Convert GPS latitude/longitude from radians to a consistent local North/East frame; include the GPS antenna offset from the estimated body point.
3. Keep the released IMU vectors in their sensor axes. Inside the estimator, apply bias/scale corrections, propagate attitude from corrected gyros, rotate corrected specific force and account for gravity. Carry attitude/calibration uncertainty jointly. Avoid an unexplained smoothing window or future-data interpolation in online inputs.
4. For FOG, unwrap the released heading and use increments/interval-average rotation with an appropriate error model. Successive differenced angle errors can be correlated; it is not an absolute-heading observation with fresh independent noise at every row.
5. For wheels, use the measured left/right speeds directly in a defined observation/kinematic model; account for scaling and departures from no-slip assumptions.
6. For GPS, check horizontal fix status, plausibility, stale/held behavior and innovations. Estimate/tune observation uncertainty on training/validation data; do not use RTK-reference error at inference.

The existing odometry mean/covariance files are **outputs of an EKF already combining wheels, FOG and IMU**. They can support replication/comparison, but should not be added as independent observations alongside their source streams. Likewise, avoid counting vendor attitude and the same IMU's raw-like measurements as unrelated sensors. For cameras, use corrected calibration downloads rather than the paper's known camera-center table typo.

## Why use these sensors, and why not every available stream?

Proposed intended full estimator: **IMU + wheel speeds + consumer GPS + FOG + vision**, with the agreed frozen SuperPoint/LightGlue frontend. Include GPS when valid; indoor/outage periods must work without it. The user explicitly asks to exploit multiple cameras for stereo-like information. Do not lock a single-view implementation by assumption: first check calibrated overlap/triangulation quality and select the initial view set. Multiple cameras do not require retraining SuperPoint/LightGlue.

There is no scientific requirement to discard a useful sensor. Build integration in stages so that each observation/noise model can be checked. Compare against an analytical baseline using exactly the same inputs. For comparison with published IMU/GPS or wheel/GPS results, reproduce or match their sensor subset separately. A gain from FOG or cameras must be reported as a sensor gain rather than attributed to a neural component. A no-FOG experiment also quantifies whether learned compensation can reduce the need for the additional gyro.

Proposed evaluation-only inputs: official pose/covariance, RTK and LiDAR reference information. The official pose is a multi-session LiDAR/RTK/odometry solution; roughly 100-Hz poses are odometry-interpolated between graph nodes. Withholding RTK/LiDAR gives a distinct deployment input budget but does **not** make this reference fully independent of IMU/wheels/FOG. Check node/reference-quality scoring before making fine-grained accuracy claims. No method receives test reference errors as an online reliability signal.

## What the closest NCLT studies actually use

| Study | NCLT estimator information | State/model emphasis | GNSS treatment |
|---|---|---|---|
| [Original KalmanNet, TSP 2022, IV-E](https://arxiv.org/html/2107.10043) | Noisy odometry velocities | Position/velocity per horizontal axis, four-state combined model | No GNSS observation in this experiment |
| [Split-KalmanNet, TVT 2023, V-B](https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/Split-KalmanNet_A_Robust_Model-Based_Deep_Learning_Approach_for_State_Estimation.pdf) | Noisy odometry velocities | Nearly constant acceleration model for planar localization | No GNSS observation described in this experiment |
| [Song et al., SPL 2024, public model](https://github.com/SongJgit/KalmanNet4SensorFusion/blob/main/Net/params/make_params.py) | Alternative IMU/GPS and wheel/GPS configurations; wheel route also receives IMU heading/rate | Six states: horizontal position/velocity, heading and angular rate; inspected model has no explicit sensor bias/scale states | Consumer receiver fixes converted to local horizontal positions; preprocessing can optionally select RTK instead |
| [KalmanFormer, 2025, Sections 4.3 and 4.5.2](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1460255/full) | IMU angular rate/acceleration and consumer GPS | Six states: horizontal position/velocity, heading and angular rate | Receiver-position observations; paper describes RTK as evaluation reference |
| [Dataset's own odometry EKF, Section 4](https://robots.engin.umich.edu/nclt/nclt.pdf) | Wheels and FOG propagation; IMU roll/pitch/angular-rate updates | Horizontal translation plus roll, pitch, yaw and three angular rates | No GPS in this odometry filter; RTK/LiDAR enter the separate reference SLAM |

The recent fusion examples use receiver fixes, not a satellite-level GNSS reconstruction. Original KalmanNet/Split's velocity-only tasks differ from the externally aided estimator we want. This is a targeted learned-filter shortlist, not a claim that all NCLT research uses only numerical sensors; NCLT also supports extensive vision/LiDAR/SLAM work.

The [Song preprocessing](https://github.com/SongJgit/KalmanNet4SensorFusion/blob/main/data/NCLT/preprocess.py) includes heading products, acceleration smoothing, fixed coordinate offsets and GPS selection/gating. Reproduction should preserve/document those choices; our independent estimator should justify its own choices rather than silently inherit them. Native-rate fusion and their reduced-rate protocol are distinct experiments.

## Agreed navigation state and proposed calibration details

**Agreed navigation task:** horizontal position, horizontal velocity and heading,

```text
navigation_state = [north_position, east_position,
                    north_velocity, east_velocity, heading]
```

**Agreed internal attitude:** retain roll/pitch for gravity compensation and sensor projection, with a coherent 3D attitude representation. This does not change the agreed planar navigation objective. A quaternion can replace the heading-only orientation representation; do not append a duplicate yaw state. Exact representation, augmented error states and camera geometry remain backend/model decisions.

A Segway tilts, and campus travel includes changes of slope. An attitude error of one degree can project approximately g sin(1 degree) = 0.17 m/s^2 into the horizontal acceleration. Thus planar navigation is compatible with tilt handling, whereas dropping two accelerometer/gyro axes and assuming roll/pitch are always zero is not a sound default. Body-axis gyro z is also not exactly world yaw rate during tilt.

**Biases describe sensors and can simultaneously be unknown estimator states.** The distinction is navigation states versus augmented nuisance/calibration states, not 'state biases or sensor biases' as competing choices. External observations correct navigation and, through the joint model/covariance, constrain persistent sensor offsets. One coherent bias quantity should appear in the sensor model; avoid separate unrestricted filter and network compensators for the same offset.

| Quantity | Starting recommendation | Reason / qualification |
|---|---|---|
| IMU yaw-relevant gyro bias | Estimate online as a slowly varying state | Direct bearing on heading drift; FOG and other observations can help constrain it |
| Horizontal accelerometer biases | Estimate online if acceleration drives propagation | Must distinguish bias from tilt error; interpretation is offset of the released filtered channels |
| Other IMU bias components | Include only as needed by the internal attitude/propagation model, with priors | Planar data do not automatically identify every axis independently |
| Wheel common speed scale | First calibration parameter to test; static train-only estimate before online/neural variation | Relevant directly to released m/s speeds; support with an observability/excitation check |
| Left/right wheel asymmetry or effective track width | Optional later calibration | Necessary only when used by the chosen wheel/yaw model; avoid estimating redundant parameters |
| FOG bias/drift | Model as nonzero uncertainty; optional slow drift state | Extra gyro is not perfect; distinguish absolute heading offset from interval rotation error |
| IMU full scale/misalignment matrices | Keep nominal/offline initially; selected gains remain candidates | Planar/limited-axis motion can be degenerate for online calibration |
| GNSS systematic position error | Begin with quality checks, robust updates and tuned uncertainty; explicit correlated bias later if justified | Free GPS bias and position states can be confounded; changing R alone does not remove a bias |
| Camera/IMU/GPS extrinsics, timing | Use supplied nominal calibration and audit first | Unrestricted online extrinsics/time offsets expand the problem and can mask model errors |

This is a proposed starting subset, not a permanent exclusion of gain learning. [OpenVINS' primary propagation/calibration documentation](https://docs.openvins.com/propagation.html) explicitly discusses sensor biases/scales and calibration degeneracies under planar or limited excitation.

No artificial 'bias of the true position state' is required: uncertainty in position is already represented by its estimate/covariance. Add a persistent physical error state only where the model gives it a defined role and observations can distinguish it.

We have not established ground-truth labels for the IMU biases, wheel gains or FOG drift. State-reference agreement can evaluate their usefulness for navigation, but does not by itself establish that an auxiliary estimate is the uniquely correct physical calibration. Horizontal velocity reference would also be derived from pose unless a separate validated reference is identified.

## Clarification: calibration stays in the sensor frame

The user's concern is valid for a pipeline that first freezes a separate orientation estimate, removes gravity, projects to two dimensions and only then tries to identify individual IMU-axis errors. A known 3D rotation itself is invertible; the difficulties come from fixed/uncertain attitude, lost correlations, filtering and discarded components.

Proposed measurement convention:

```text
measured_accel = D_a * true_specific_force + accel_bias + accel_noise
measured_gyro  = D_g * true_angular_rate   + gyro_bias  + gyro_noise
```

All vectors above use the released sensor axes; D is a multiplicative gain/misalignment matrix. The estimator uses

```text
corrected_force = inverse(D_a) * (measured_accel - accel_bias)
corrected_rate  = inverse(D_g) * (measured_gyro - gyro_bias)
navigation_acceleration = R_sensor_to_navigation(attitude) * corrected_force
                          + navigation_gravity
```

Bias/gain parameters are thus expressed in their original released sensor axes even though the navigation state uses a local frame. They can be estimated jointly or supplied by a coherent learned parameter model. The same physical bias should not be independently compensated twice.

For fixed current D and attitude, the accelerometer's transformed noise contribution is

```text
A = R_sensor_to_navigation * inverse(D_a)
transformed_accel_noise_covariance = A * sensor_accel_covariance * transpose(A)
```

The joint estimator must additionally carry attitude/parameter uncertainty and cross-covariances through its propagation Jacobians. The displayed expression is not the complete navigation covariance. Gyroscope noise is treated in its own axes and mapped through attitude propagation. Calibration before transformation does not establish that all calibration parameters are identifiable from planar motion. [Primary measurement/propagation reference](https://docs.openvins.com/propagation.html).

NCLT's internal device filtering cannot be undone. The parameters above describe residual errors of the released sensor-frame channels, not a recovered pre-factory-calibration ADC signal.

FOG can be included in the same **inertial subsystem** as the Microstrain IMU, sharing the vehicle attitude while retaining a separate bias, scale, covariance, timing and mounting model. It is physically a second single-axis gyro, not a complete additional IMU. For its unwrapped angle increment along the sensing axis, a schematic correction is

```text
corrected_FOG_rotation = (unwrapped_angle_change - FOG_rate_bias * dt)
                         / FOG_scale
```

FOG_rate_bias has angular-rate units and FOG_scale is dimensionless. The prediction must use rotation about the mounted FOG axis, rather than assuming its angle is absolute world yaw. Model increment noise and any relevant correlations; do not interpret the integrated angle's arbitrary zero as a gyro-rate bias.

## FOG-assisted inertial propagation and recorded timing

The user has now agreed to **one analytical inertial subsystem combining IMU and FOG, followed by physical navigation propagation**, with wheel/GNSS/visual observations. FOG need not be treated only as a downstream pose correction. Exact interval fusion and joint noise treatment remain open. NCLT's own wheel-driven odometry already uses FOG rotation in its propagation, although our acceleration-driven model is different. [Original model, Section 4](https://robots.engin.umich.edu/nclt/nclt.pdf).

Conceptually:

1. Correct the released IMU accelerometer/gyro and FOG measurements in their own axes.
2. Align their rotation information over compatible causal time intervals and transform the known mounting axes.
3. Combine redundant angular information analytically using its uncertainty; FOG supplies one axis, while IMU gyros supply all three.
4. Propagate attitude using the combined rotation, then velocity/position using corrected specific force and gravity.
5. Apply wheel, receiver-GNSS and visual updates at their own timestamps.

For illustration only, two independent unbiased rotation estimates about the same axis and over the same interval have the scalar fusion rule:

```text
FOG_weight = IMU_rotation_variance / (FOG_rotation_variance + IMU_rotation_variance)
combined_rotation = FOG_weight * corrected_FOG_rotation
                    + (1 - FOG_weight) * corrected_IMU_rotation
```

The actual joint estimator must carry calibration uncertainty, timing and relevant cross-correlations. This illustrative rule is not a locked implementation. It is not valid to add the two rotations as successive physical motions or to fuse a FOG-containing propagation and the same FOG data as independent evidence. IMU axial gyro data remain useful for redundancy, fallback and relative calibration; estimating both sensors' biases/scales needs an observability analysis and external constraints, not just their agreement.

Timing audit repeated on the complete **2013-01-10** sensor archive:

| Stream | Measured logged timing | Scope |
|---|---|---|
| FOG | 97,007 rows; mean 94.54963 Hz; median interval 9.998 ms; mean interval 10.57646 ms | Approximately 100-Hz cadence with irregular arrival/logging intervals; not a constant clock |
| Microstrain vectors | Mean 47.099 Hz | Same inspected recording; not a guarantee for every day |

FOG interval fifth/ninety-fifth percentiles are 0.126/19.906 ms. Very short intervals make naive per-row angle differentiation a poor default; audit batching/timing before assigning a physical instantaneous rate. Integrating over compatible intervals can avoid amplifying this timing behavior. The paper says FOG timestamps use message arrival time. Do not substitute the hardware manual's output capabilities or the released odometry's 100-Hz cadence for the actual input timestamps. [Inspected archive](https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/sensor_data/2013-01-10_sen.tar.gz), [timestamp policy](https://robots.engin.umich.edu/nclt/nclt.pdf).

## Clarification: Ladybug stereo and multiple views

Stereo-like triangulation is possible in overlap regions when rays have useful intersection geometry. The Ladybug's surrounding views do not behave like six cameras all looking at the same scene patch. [Schneider and Förstner, 2013, Section 4.3.2](https://www.dgpf.de/pfg/2013/pfg2013_4_schneider.pdf) specifically describes the small overlap of neighboring Ladybug3 views. The [manufacturer manual](https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/manuals/Ladybug3-Technical-Reference.pdf) lists a nominal 48-mm side-camera baseline.

From the [corrected NCLT calibration ZIP](https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ladybug3_calib/cam_params.zip), distances between adjacent side-camera centers are **48.98–49.47 mm**, computed from the supplied translations. This is a calibration calculation, not a measured triangulation-accuracy result. Short baselines give weak depth information for distant points. Extra views can still help tracking coverage and rotation constraints without producing accurate instantaneous depth for every feature.

The subsequent pair-distance check gives non-neighboring side-camera baselines of **79.52–79.78 mm**, with cam1–cam4 the largest at 79.78 mm. Greater physical separation is not an optimal selection rule without useful shared views and favorable triangulation geometry. Two suitable views already triangulate 3D feature coordinates, including relative height; a third camera is an information/coverage choice, not a prerequisite for vertical depth. Begin with two cameras and verify the actual pair using a small training-only image/calibration audit. One-to-six-camera/all-pairs support remains a design capability. See [visual configuration and selection rationale](./visual-operating-configuration.md#why-the-farthest-cameras-are-not-automatically-best).

Agreed design capability: run with any selected set of **one through six cameras**, including an **all-pairs matching option** and a reduced useful-pair option for compute/latency comparisons. Six cameras have 15 distinct simultaneous unordered pairs. Nonoverlapping or poorly conditioned pairs need not yield accepted geometric observations. Extract/cache features once per image and reuse them across matchers. Temporal tracks remain available even when only one camera is selected.

Use original individual views with calibrated geometry, masks and frozen SuperPoint/LightGlue temporal/cross-camera tracks. Check epipolar consistency, parallax, reprojection errors and coverage; combine shared tracks coherently in the locked MSCKF-style update. Do not treat a stitched panorama as a perfect single-center image or fuse derived poses sharing images as independent evidence. No new visual-network training is required; multi-camera support remains engineering work. Default camera/pair schedule and measured runtime remain open.

Moving points can match and triangulate correctly in a simultaneous camera pair while violating the static-landmark model across time. **Frozen DynaSLAM-style instance masking, RANSAC, IMU/FOG-prior motion checks and calibration-consistency protection are now baseline choices.** Mask R-CNN is the first reference; exact pretrained implementation/cadence remains open. The [dynamic-scene architecture discussion](./architecture-options.md) gives the combined policy. LightGlue confidence alone is not a stationarity test.

## Clarification: sensor dropout and comparisons

Requested design capability: modular sensor availability. For an unavailable **observation**, omit its update and continue propagation with uncertainty growth. A missing **propagation input** requires a defined fallback: an IMU-driven filter cannot handle IMU loss solely by skipping a measurement update. Wheels/FOG may support an alternative propagation mode; that mode must be specified and tested.

Agreed diagnostics: reject unsupported startup configurations or missing indispensable propagation with no fallback; warn and report degraded mode/uncertainty when temporary loss weakens constraints but propagation remains valid. Normal temporary GNSS outages should not automatically stop the estimator. Distinguish the requested navigation outputs from optional calibration observability: weak gain excitation can freeze parameter adaptation rather than terminate navigation. Configuration checks plus runtime conditioning/innovation/uncertainty diagnostics are useful; a sensor-presence list alone cannot prove observability for every motion or visual scene.

Sensor removal can change observability, initialization, metric scale and bias/gain identifiability. Vision-only monocular operation, for example, does not generally retain observable metric scale. Missing-input masks, observation age and appropriate training/validation are needed for learned modules; zero-filling a missing sensor does not by itself create a valid model for that subset. Distinguish transient loss from a model trained/deployed permanently without that sensor.

Modularity enables matched sensor-subset experiments. It does not make every subset equally accurate or automatically reproduce another publication. Comparisons still need matched released/derived inputs, initialization, splits, timing, alignment, reference and metrics. These are design/evaluation requirements, not a claim that an implemented dropout-capable estimator already exists.

## Clarification: what is the ground truth?

Proposed main reference: the official groundtruth_DATE.csv **six-DOF body pose**, x/y/z/roll/pitch/yaw, in the dataset's common local North-East-Down frame. It comes from a batch, multi-session SLAM graph with LiDAR scan matching, high-grade RTK and odometry. It is not merely the consumer Garmin fix, and not the standalone gps_rtk.csv trajectory. [Dataset paper, Section 6](https://robots.engin.umich.edu/nclt/nclt.pdf).

The official site supplies keyframe marginal covariances and clarifies that dense approximately 100-Hz poses are odometry-interpolated between graph nodes. Proposed evaluation: primary horizontal position/heading checks at supported graph nodes with quality/covariance inspection, plus a dense-reference sensitivity check. The nodes still share some odometry information; node-only scoring reduces interpolation dependence without creating an independent physical reference. Bias/gain truth remains unavailable, and reference velocity is derived unless a separate validated source is identified. [Official clarification](https://robots.engin.umich.edu/nclt/).

## Agreed items and next discussion

- Agreed: use released wheel speeds and consumer-GNSS receiver solutions; no raw satellite solver.
- Agreed: frozen SuperPoint/LightGlue with selectable one-to-six-camera operation, all-pairs and reduced-pair modes; actual frontend/backend performance remains untested.
- Agreed: skip barometer.
- Agreed: magnetometer disabled initially and potentially indefinitely; no required later experiment.
- Agreed: modular sensor removal with warnings/errors appropriate to supported propagation and observable outputs.
- Agreed: IMU/FOG-assisted inertial propagation with wheel/GNSS/visual updates, preserving separate physical sensor/error models and consistent joint uncertainty. Exact fusion mathematics remains open.
- Agreed: five reported quantities—north/east position, north/east velocity and heading—and tilt-aware 3D internal attitude.
- Agreed sensor-error categories: selected IMU biases/scales, FOG bias/scale and wheel calibration. Exact axes, augmented-state ownership versus neural priors/dynamics, initialization and cadence remain open.
- Agreed learning roles: sensor adaptation, model residual and uncertainty adaptation. This does not require three separate networks or every possible output enabled.
- Presentation preference: use plain-text equations/code blocks, not LaTeX delimiters, in chat and maintained notes.

**Backend locked:** error-state EKF with tightly coupled MSCKF-style visual updates, bounded recent rig poses and no persistent map/loop closure initially. Pilot ownership, neural clocks/memory, startup/recovery, baseline mathematics, initial numerical/visual settings and training goals with event-aware regularization are accepted. Training protocol is accepted. Evaluation/comparisons and implementation/compute are accepted. Rollout order is accepted. **Current discussion:** [model learning and Gemma 4](./model-learning-and-gemma4.md). Start with two cameras and audit overlap/triangulation for the exact pair. Investigate the soft lateral no-slip constraint subsequently within this project; reserve deeper linearization/data-driven modelling for the final design discussion.
