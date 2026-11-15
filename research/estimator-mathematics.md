# Detailed estimator mathematics — accepted baseline

14 November 2026. **Baseline recommendations accepted by the user.** Startup/recovery, the ESKF/MSCKF backend and pilot network outputs are already agreed. The soft lateral no-slip constraint is committed subsequent work within this project, after the forward-speed pilot. The accepted baseline linearization policy will receive a dedicated deeper discussion together with data-driven modelling as the final design decision. Numerical tuning and exact implementation derivations remain engineering work.

## Internal state and conventions

Recommend a standard 3D inertial core, with five planar reported outputs:

```text
Internal navigation: position(3), velocity(3), attitude quaternion
Persistent errors:   accelerometer bias(3), gyro bias(3), FOG rate bias(1)
Visual history:      bounded clones of position(3) and attitude
Reported output:    North/East position, North/East velocity, heading
```

The core has 16 error coordinates before clones: 3 position + 3 velocity + 3 attitude error + 3 accelerometer bias + 3 gyro bias + 1 FOG bias. Quaternion storage has four entries; its error has three. Calibration gains remain fixed in the pilot; later stages augment the contract explicitly.

A strictly planar internal model is smaller but would require additional approximations for tilt, slopes and camera geometry. Recommend retaining height/vertical velocity as nuisance states rather than enforcing constant world height. This does not expand the reported accuracy objective or establish that every calibration state is observable.

Use a local North-East-Down world frame. Let R map body axes into world axes; body axes are forward/right/down. Locate the navigation position/velocity at the IMU, with body-aligned orientation, and transform the reported trajectory to the documented axle-center reference point. Apply all sensor mounting rotations/lever arms explicitly. Biases remain in each released sensor's axes before rotation; gravity is handled by the estimator, not removed with the vendor attitude filter.

For a new derivation use a right-multiplicative rotation error, `R_true = R_est * Exp(skew(attitude_error))`, with quaternion nominal storage. If adapting OpenVINS, convert/rederive consistently: its left-error/JPL convention is different. Copying its Jacobians into another convention without conversion is invalid. [Solà's ESKF reference](https://arxiv.org/abs/1711.02508), [OpenVINS conventions](https://docs.openvins.com/fej.html).

## Inertial propagation

Recommend midpoint integration over completed intervals using actual timestamps, rotation-exponential updates and a consistent discrete error/noise model. Scale/misalignment matrices are fixed/offline in the pilot. Schematically:

```text
specific_force_body = mounting_rotation * calibrate(accel_sensor - accel_bias)
angular_increment   = analytical_IMU_FOG_fusion(completed_interval)
R_next              = R * Exp(skew(angular_increment))
acceleration_world  = R_mid * specific_force_body_mid + gravity_NED
position_next       = position + velocity * dt + 0.5 * acceleration_world * dt^2
velocity_next       = velocity + acceleration_world * dt
IMU_bias_next       = IMU_bias + learned_bias_drift(history) * dt
FOG_bias_next       = FOG_bias                     # with fixed random-walk noise
P_pred              = F * P * transpose(F) + Q_discrete
```

Here `F` is the discrete transition Jacobian, including calibration-state dependence; `Q_discrete` is obtained from baseline physical noise over that interval, not copied unchanged across variable dt. Clones retain their cross-covariances with the current navigation/calibration state. The displayed acceleration formula is a midpoint approximation, not a claim of exact integration. [OpenVINS inertial measurement models](https://docs.openvins.com/propagation.html).

### FOG within the inertial subsystem

Recommend common-interval angular fusion of corrected IMU gyro information and FOG increments. The released FOG channel is integrated relative angle: unwrap and difference it over compatible causal intervals, subtract its rate bias times dt and apply fixed pilot scale/mounting. It measures rotation about its mounted sensing axis, not absolute world yaw. Retain IMU information on all axes, with IMU-only fallback when FOG is absent.

Use covariance-weighted analytical fusion conditional on calibration states. Carry those states' uncertainty through the joint propagation Jacobian; do not fold it into input noise and count it a second time. Account for shared interval boundaries, timestamp alignment and possible temporal correlation in differenced angle errors. If the selected model needs a short colored-noise augmentation, include it explicitly. Exact fusion/noise discretization is an implementation derivation to verify against the released channel, not a claim that a naive independent weighted average is already sufficient. Do not also fuse the same FOG increment as an independent heading update. Relative gyro/FOG agreement alone cannot identify both absolute biases/scales. See the [audited FOG contract](./nclt-sensors-and-state.md#fog-assisted-inertial-propagation-and-recorded-timing).

## Observation relations

| Input | Accepted observation model | Pilot learning |
|---|---|---|
| Wheel speeds | Scalar mean of calibrated left/right speeds, predicted by forward body speed at the wheel/axle reference point; include rotational lever-arm velocity | Add the locked scalar wheel-model residual to the prediction; scale baseline observation variance |
| GNSS fix | Horizontal antenna position from IMU position plus rotated antenna lever arm in the local frame | Separate positive North/East baseline-variance multipliers; no new GNSS bias state |
| Feature tracks | Calibrated 3D projection into each camera from the corresponding retained pose and temporary landmark geometry | One multiplier per accepted original feature-observation 2x2 noise block |

Wheel convention, with the lever arm pointing from the IMU to the axle center:

```text
velocity_axle_body = transpose(R) * velocity_IMU_world
                     + cross(angular_rate_body, lever_arm_IMU_to_axle)
wheel_prediction  = forward_component(velocity_axle_body) + learned_wheel_error
wheel_innovation  = mean(left_speed, right_speed) - wheel_prediction
GNSS_prediction   = horizontal(position_IMU + R * lever_arm_IMU_to_antenna)
```

Using an angular-rate measurement in the wheel lever-arm correction introduces input-noise dependence shared with propagation. Include its contribution and relevant cross-correlation, or justify a bounded approximation; do not silently treat it as an unrelated exact number. Reported axle-center velocity uses the same geometry.

Use **forward wheel speed only in the initial pilot**, without a hard lateral or world-vertical zero-velocity constraint. **Implement and investigate a soft lateral no-slip constraint in a subsequent stage of this project.** This is committed project work, not optional future work. Compare it on/off under matched inputs and investigate covariance inflation/rejection during suspected lateral sliding. This preserves the locked pilot observation dimension while explicitly scheduling the additional ground-motion relation.

For vision, whiten with the accepted observation covariance, linearize multi-view reprojection, eliminate the temporary landmark by nullspace projection and compress as appropriate before the analytical update. With unequal learned observation noise, transform/whiten the covariance consistently; orthogonal projection does not preserve an arbitrary covariance as an identity matrix. Consume each observation's information once. Masking/RANSAC/inertial checks precede the accepted update; a verification pose is not a second independent measurement. [MSCKF nullspace derivation](https://docs.openvins.com/update-null.html).

## Covariance, linearization and learned uncertainty

Recommend an ordinary joint error covariance with a Joseph-form update, followed by nominal-state injection and the corresponding error-coordinate covariance reset. Use linear solves rather than an explicit matrix inverse. Preserve navigation/bias/clone cross-covariances. Pilot process noise remains fixed; learned observation multipliers scale their calibrated baseline blocks.

```text
innovation = measurement - predicted_measurement
S          = H * P_pred * transpose(H) + R_measurement
K          = solve_for_gain(P_pred, H, S)
error      = K * innovation
P_updated  = (I-KH) * P_pred * transpose(I-KH)
             + K * R_measurement * transpose(K)
nominal    = inject(nominal_pred, error)
P_reset    = reset_Jacobian * P_updated * transpose(reset_Jacobian)
```

These are the ordinary formulas for an observation noise independent of the predicted error. Where input/measurement reuse induces correlation, use the corresponding correlated-noise update/augmentation rather than silently applying this independent-noise form. Symmetrization handles roundoff, not a wrong covariance model. A square-root implementation can be considered if conditioning becomes a demonstrated issue.

Use observability-aware/FEJ treatment for the visual-inertial subsystem, adapted to GNSS/wheel/FOG and their actual observable directions. Differentiate explicit state dependence of the combined physical and learned mean models; do not insert an independently learned arbitrary Jacobian. [OpenVINS FEJ explanation](https://docs.openvins.com/fej.html).

The user accepts this as the baseline policy and requests a deeper discussion **as the final design decision**, together with data-driven modelling. That discussion will distinguish nonlinear model mismatch from linearization error, consider learned residual/submodels/physical parameters and their identifiability, and decide which additional approaches to investigate within this project. No enhanced linearization or data-driven model architecture is locked by accepting the baseline.

For the pilot, recommend conditioning on the causal GRU memory instead of augmenting every hidden unit into the filter covariance. This is an explicit approximate adaptive EKF: hidden-memory uncertainty, selection/gating effects and neural cross-time errors are not fully represented. Positive Q/R does not remove that limitation. Check innovation/trajectory consistency and uncertainty calibration against the fixed-noise ablation; revisit augmentation or a richer error model if required. Current-observation-dependent noise heads are adaptive robust weighting, not independent sensors.

## Scope of this decision

The accepted baseline fixes the internal 3D nuisance state, conventions, integration/model family, observation relations and uncertainty policy. Exact FOG colored-noise derivation, analytic/autodiff Jacobians, numerical noise/gates, cloning schedules and implementation tests are engineering work to verify as we build. The lateral constraint is scheduled within this project; linearization/data-driven modelling is the final dedicated design discussion. No estimator has been implemented or evaluated. Next decision group: [numerical constraints and a small initial network configuration](./calibration-ownership.md#numerical-constraints-and-network-configuration--working-settings).
