# Benchmark data and novelty audit

Research date: 13 November 2026. Purpose: answer whether the proposed architecture is novel, identify the real datasets used by the recent KalmanNet-related papers, and distinguish pooled training from repeated experiments. This is a targeted audit of our shortlist, not a field-wide publication census. No estimator or data archive has been run/downloaded in this pass.

**Subsequent user decisions:** **NCLT is locked** as the first development/training/evaluation dataset. VID indoor and the 2024 racing-car data are optional future investigations. Novelty is **not required**; retain the assessment below as historical related-work context, not a reason to redirect the project or an unfinished gate. Current model/parameter discussion: [NCLT modeling options](./nclt-modeling-options.md).

## Novelty assessment

**The broad architecture is not an established novelty claim.** Keeping a physical model, learning a model residual or calibration/noise component, and retaining a mathematical estimator update already has close prior art. Our exact state, nominal model, learned outputs, constraints, losses and inference contract are not yet fixed, so an exact-method novelty judgment is premature.

| Proposed ingredient | Close prior art / overlap | Consequence |
|---|---|---|
| Physical drone dynamics plus learned residuals inside visual-inertial estimation | HDVIO (RSS 2023) and HDVIO2.0 (T-RO 2025) combine physical dynamics with learned aerodynamic corrections and estimate robot state/disturbances. [HDVIO](https://arxiv.org/html/2306.11429), [HDVIO2.0](https://arxiv.org/html/2504.00969) | The overall residual-physics/VIO pattern is established. A different backend or feature extractor alone is not a sufficient methodological claim |
| Learned dynamics/noise with an analytical Kalman correction | Węgrzynowski et al. (IROS 2024) investigate physical, residual and learned tire models in a differentiable UKF, optimize noise parameters, and test state-dependent Q/R and online friction estimation. [Paper](https://arxiv.org/html/2408.15610) | Even the broad combination of model learning and uncertainty adaptation with analytical gain already exists |
| Frozen learned image features/matching with conventional estimation | Practical Deep Feature-Based VIO (2024) combines SuperPoint–LightGlue tracking with VINS-Mono. [Author record](https://researchportal.rma.ac.be/en/publications/practical-deep-feature-based-visual-inertial-odometry/) | Our selected visual route is a component choice; it does not establish novelty |
| Bias/scale calibration during motion | Classical visual-inertial self-calibration and learned IMU calibration such as Calib-Net. [Observability analysis](https://arxiv.org/html/2201.09170), [Calib-Net](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.772583/full) | Selecting identifiable gains and an effective adaptation method needs a narrower comparison |
| A specific joint constrained scale/bias/residual/uncertainty method | Exact algorithm and evaluation contract remain undefined | No claim that nobody has combined the exact components; this requires a focused search after the method is specified |

This does not make the project uninteresting. Its north star allows a demonstrated accuracy, compute, latency/history or sensor advantage. A worthwhile contribution could be a specific estimator design, uncertainty/identifiability analysis, or a well-supported deployment tradeoff. Those are research hypotheses; we have not established the gap or the advantage yet. Merely assembling components that have not appeared in exactly the same diagram is insufficient evidence of a new scientific method.

The KalmanNet family is a useful learned-filter comparison family, while HDVIO and the analytical-UKF vehicle work offer close physical-model design ideas. The previously deferred model-matrix/linearization discussion has now resumed in the [NCLT modeling note](./nclt-modeling-options.md).

## What the KalmanNet-related papers actually train and test on

**NCLT is the most repeated named real-world dataset in this reviewed family.** Six named uses are established below, including original KalmanNet and the abstract-level MM-KalmanNet record. This is a lower-bound count in this shortlist, not a count of all publications benchmarking on NCLT. Cholesky/MAML records with incomplete experimental access are not used to infer additional dataset counts.

| Study / inspected source | Real benchmark data | Training and evaluation protocol established by this audit | Multiple named real training datasets? |
|---|---|---|---|
| Original KalmanNet, TSP 2022 | **NCLT**; additional synthetic systems | Real experiment uses one recording, 2012-01-22, split approximately 85/10/5% for train/validation/test at 1 Hz. Observations are odometry velocity. [Section IV-E](https://arxiv.org/html/2107.10043) | No pooled real-dataset training shown in this experiment |
| Split-KalmanNet, TVT 2023 | **NCLT**; circular-motion simulations | Train/validate on 2012-01-22 and test on 2012-04-29, at 1 Hz; noisy odometry velocity supplies the real-data observations. [Sections V-A/B](https://www.weizmann.ac.il/math/yonina/sites/math.yonina/files/Split-KalmanNet_A_Robust_Model-Based_Deep_Learning_Approach_for_State_Estimation.pdf) | One named real dataset, different recording days |
| Song et al., integrated-navigation KalmanNet, SPL 2024 | **NCLT** | The current official preprocessing code defines 22 recording days for training, 2 for validation, and 3 for testing. Supports IMU/GPS and wheel/GPS variants; the inspected wheel input also includes IMU heading/rate. Data is prepared for 1 Hz filtering. [Official preprocessing](https://raw.githubusercontent.com/SongJgit/KalmanNet4SensorFusion/main/data/NCLT/preprocess.py), [parameters](https://raw.githubusercontent.com/SongJgit/KalmanNet4SensorFusion/main/Net/params/make_params.py) | One named dataset, many sessions; do not assume one monolithic raw IMU/wheel/GPS fusion design from the abstract alone |
| KalmanFormer, published January 2025 | **NCLT**; additional simulations | IMU and consumer GPS; train/validate on 2012-01-22, test on 2012-04-29, at 1 Hz. Describes RTK GPS as the reference. [Sections 4.3 and 4.5.2](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1460255/full) | One named real dataset |
| Enhanced KalmanNet, Signal Processing 2026 | **NCLT** | Publisher section excerpts specify GPS/IMU/wheel localization, with 22 training, 2 validation and 3 test trajectories from the 27 sessions. Exact dates and full preprocessing have not been verified. [Publisher excerpts](https://www.sciencedirect.com/science/article/abs/pii/S0165168425005730) | One named real dataset in the documented localization experiment |
| MM-KalmanNet, CVCI 2025 | **NCLT plus separately collected localization measurements** | Publisher abstract establishes IMU/GNSS and evaluation on both sources; the training mix/splits were not recovered. [Publisher abstract](https://ieeexplore.ieee.org/abstract/document/11348299) | **Unverified**; two evaluation sources do not establish pooled training |
| Adaptive KalmanNet, ICASSP 2024 | No real public dataset in the inspected paper | Generated linear systems with Gaussian/non-Gaussian noise; trains on a small number of noise settings and tests interpolation/shifts. [Paper](https://arxiv.org/html/2309.07016) | Synthetic noise conditions, not several real datasets |
| Bayesian KalmanNet, TSP 2025; inspected arXiv v4 July 2026 | No real navigation dataset in that inspected experimental section | Generated linear, pendulum and 2D constant-velocity navigation systems. Learns across noise levels within the experimental setup; its navigation case uses 100 generated trajectories with 5 validation and 10 test trajectories. [Sections V-A/V-D](https://arxiv.org/html/2309.03058) | Synthetic examples; not evidence of pooled real-dataset training. Latest preprint version is explicitly identified |
| Recursive KalmanNet, EUSIPCO 2025 | No real public dataset in the inspected experiment | Generated 1D position/velocity tracking with position observations and bimodal noise; 1000 train, 100 validation, 1000 test sequences, 150 samples each. [Section IV](https://arxiv.org/html/2506.11639) | One synthetic model family, with noise variations |
| Latent-KalmanNet, TSP 2024; inspected 2023 preprint | No real navigation dataset in the inspected experiments | Synthetic image observations for pendulum/Lorenz experiments; evaluates accuracy, noise robustness and trajectory-length generalization. [Section IV](https://arxiv.org/html/2304.07827) | Multiple synthetic tasks do not demonstrate a shared multi-real-dataset pretrained navigation model |
| RTSNet, TSP 2023 | No real navigation dataset in the inspected experimental section | Generated linear/kinematic systems, Lorenz and Van der Pol. These are smoothing experiments with future observations. [Section IV](https://arxiv.org/html/2110.04717) | Several synthetic experiments, not a demonstrated pooled real-data navigation training recipe |
| Unscented KalmanNet, August 2026 preprint | **UZH-FPV** plus three synthetic examples | 11 indoor flights; leave-one-flight-out, train on the other 10. Each inspected flight segment is 27 seconds. Propagation and pseudo-velocity both use the IMU; no independent visual/GNSS correction in this flight experiment. [Section 4.4](https://arxiv.org/html/2608.04201) | One named real dataset with many train/test folds |
| Cholesky-KalmanNet, SPL 2025 volume | Abstract/manuscript excerpts report synthetic and real tests | Full dataset/split inventory remains **unverified** due to inaccessible full experimental text in this pass. [Manuscript](https://par.nsf.gov/servlets/purl/10656999) | Do not infer either NCLT or pooled training from related papers |
| MAML-KalmanNet, TSP 2025 | Bibliography identified; primary experimental text not recovered | Training and real-dataset inventory remain **unverified** in this review | Not counted |

## Multiple recordings, datasets, systems and pretrained components

These are distinct situations:

1. **Many recordings from one named dataset:** the 22-session NCLT training recipe is one dataset, with multiple journeys and conditions. It does not demonstrate generalization to a new robot or campus.
2. **Several synthetic systems in a paper:** each experiment can have its own generated training/test data and trained estimator. This is not evidence that one model was pretrained jointly on all systems and transferred to a real robot.
3. **Several named real evaluation datasets:** this does not establish pooled training or zero-shot transfer. Inspect which weights are trained for each platform and experiment.
4. **Upstream pretrained vision:** even if our estimator module is trained only on NCLT or VID, frozen visual weights bring earlier training information. Original SuperPoint uses Synthetic Shapes and COCO; LightGlue uses homography pretraining and MegaDepth. We would disclose this, while keeping our estimator's own train/test sessions separate. [SuperPoint training](https://arxiv.org/html/1712.07629), [LightGlue training](https://arxiv.org/html/2306.13643).

No need has been established to pool several named real datasets for the first estimator experiment. A good sequence/session split is more important than counting short overlapping windows as independent data. Cross-dataset validation would support a stronger claim later, but does not have to be the initial training recipe.

## Closer physical-model papers use a different dataset family

| Study | Data and training | Relevance |
|---|---|---|
| HDVIO, RSS 2023 | NeuroBEM, Blackbird, VID and newly collected wind experiments, with platform/experiment-specific training. VID indoor/outdoor have separately trained models. [Paper](https://arxiv.org/html/2306.11429) | Close drone residual-dynamics state-estimation prior art |
| HDVIO2.0, T-RO 2025 | NeuroBEM for force/model evaluation; Blackbird and VID for trajectory estimation; additional wind/closed-loop experiments. VID uses separate indoor/outdoor models because vehicle mass changes. Indoor training/validation uses 6 minutes without external perturbations (80/20 split); outdoor uses separate training data and supervision. [Paper](https://arxiv.org/html/2504.00969) | Several datasets in the paper do not imply one pooled model; NeuroBEM force prediction is distinct from full VIO benchmarking |
| Węgrzynowski et al., IROS 2024 | One introduced F1TENTH-family RC-car dataset: IMU, RPM/current/steering and OptiTrack pose; velocities derived from pose. Trains on tire A with 70/20/10 split and evaluates transfer to tire C. [Paper](https://arxiv.org/html/2408.15610) | Close analytical-UKF/learned-model/uncertainty prior art; outputs dynamics states rather than global navigation pose |

Thus **NCLT dominates the named KalmanNet navigation examples**, while **VID/Blackbird are better-aligned precedents for camera-aided drone actuator dynamics**. The most reused dataset in one algorithm family is not automatically the best physical-data contract for every proposed module.

## NCLT as a concrete comparison-first candidate

NCLT is a **Segway ground-robot** dataset, not a passenger-car dynamics dataset. Its 2016 IJRR publication describes recordings from 2012–2013; the official site lists 27 sessions, about 34.9 hours and 147.4 km. It provides consumer GPS, RTK GPS, wheel data, IMU, a fiber-optic gyro, omnidirectional cameras and LiDAR. [Official dataset](https://robots.engin.umich.edu/nclt/).

### Sensor and reference contract

- A relevant first task is GNSS-aided planar navigation using IMU and/or wheels, with a specific fused input set defined explicitly. Consumer GPS provides independent spatial information beyond inertial/odometric propagation. Images may be added, but are not required by those recent KalmanNet navigation examples.
- The official reference is **offline multi-session SLAM using LiDAR scan matching, high-grade RTK GPS and odometry**, with odometry-based interpolation between graph nodes. It is not motion capture and is not completely independent of all candidate inputs. Withhold LiDAR and RTK from the evaluated sensor subset, document odometry sharing and use reference quality information where available. [Dataset paper, section 6](https://robots.engin.umich.edu/nclt/nclt.pdf).
- Reference choice varies across papers: KalmanFormer describes RTK as reference, whereas Song et al.'s current preprocessing loads the official ground-truth pose files. Results cannot be compared as if these references and valid intervals were identical. A shared rerun protocol resolves this.
- The provided learned-filter example prepares a 1 Hz task, initializes from reference pose and uses filtered IMU/heading inputs. These conventions must be matched in a reproduction and reported honestly. A higher-rate causal experiment is a subsequent protocol choice, not the same published benchmark.
- The listed navigation channels do not establish a command/RPM/torque actuator-identification experiment. A sensor-scale/bias/kinematic residual study and an actuator-dynamics study have different input requirements.
- SuperPoint–LightGlue remains the selected camera route. Beginning NCLT with numerical sensors would be a proposed change in experiment order, not evidence that the user has discarded camera experiments. Ladybug3 is a 5 Hz omnidirectional rig, not the ordinary rectified stereo setup discussed earlier. Its image archives are much larger than navigation-sensor archives.

The official January 22 listing is 107 MB for navigation sensors and 103 MB for the reference, versus 92 GB for its image archive. This is listed metadata, not a successful payload download. [Official download table](https://robots.engin.umich.edu/nclt/).

### Reusable session-level split

The currently published Song et al. preprocessing defaults are:

- **Validation:** 2013-01-10, 2013-02-23.
- **Test:** 2012-11-16, 2013-04-05, 2012-11-04.
- **Training:** the other 22 listed sessions.

These dates were read from the public source, not guessed from the 22/2/3 counts. Enhanced KalmanNet reports the same counts, but its exact dates were not verified. Do not assume they are identical. [Official preprocessing as inspected](https://raw.githubusercontent.com/SongJgit/KalmanNet4SensorFusion/main/data/NCLT/preprocess.py).

## Dataset decision — resolved

**NCLT is selected first by explicit user instruction.** The session split, exact sensor/state/model contract and comparators remain to be fixed. A numerical GNSS/IMU/wheel starting experiment is a recommendation, not an agreed removal of cameras. Dataset-name equality alone does not establish equivalent experiments.

**Future-work candidates:** VID indoor and the 2024 racing-car dataset may be investigated later; neither is required. Existing VID access/data-volume checks remain in the [decision record](./project-decisions.md).

The earlier dataset-priority question is resolved by the user's NCLT selection. Do not ask it again. No dataset archive download or implementation has started. Novelty is not a required next step.
