# Visual operating configuration — accepted starting settings

14 November 2026. **Starting recommendations accepted, with camera-pair selection still to verify.** Begin with two cameras; a third is a possible measured extension. One-to-six-camera and all-pairs operation remain design capabilities rather than the starting workload. Other resolution, feature/update budget, checkpoint, mask cadence, pose-window and caching recommendations are accepted adjustable defaults. The user asks whether maximizing physical baseline is optimal; the geometric recommendation below requires useful overlap and triangulation quality. No image sequence, segmentation/matching workload or M5 runtime has yet been audited/run.

## Accepted initial settings and camera-selection recommendation

| Choice | Initial recommendation | Rationale / alternative |
|---|---|---|
| Camera subset | **Two initially**, selected for useful overlap, triangulation and temporal tracking; neighboring horizontal views are the first candidates | The user favors a large baseline. Maximize useful triangulation information subject to view overlap, static matches and calibration, rather than center distance alone. Exact IDs and the value of a third camera are implementation-audit questions |
| Matching graph | Temporal matching within each selected camera, plus the simultaneous cross-camera edge when geometry/overlap permits | Gives temporal motion constraints without requiring useful stereo depth everywhere. For two cameras this means two temporal edges plus one spatial edge per new rig frame |
| Frame cadence | Use the released approximately 5-Hz image stream initially | Avoid an additional initial frame-decimation decision. Record true acquisition times, missing cameras and processing delay; compatible synchronized images share a rig clone |
| Working resolution | Longest image edge 800 pixels, aspect ratio preserved | Adjustable first accuracy/compute tradeoff. Apply calibrated image orientation/rectification consistently and transform intrinsics for the exact resize/crop |
| Keypoint budget | Up to 1024 retained background keypoints per image, spatially distributed | Generate up to 2048 candidates before instance-mask filtering if needed. Masking can leave fewer valid points; do not refill from suspect foreground merely to meet a quota |
| Matcher | Frozen SuperPoint-trained LightGlue, initially with reference adaptive defaults | No new training or frontend sweep required. Keep keypoint identities coherent and reuse each image's extraction across match edges |
| Segmenter | Frozen torchvision Mask R-CNN ResNet-50-FPN, explicit COCO_V1 weights, evaluation mode | Concrete initial implementation of the agreed masking baseline; not a claim it is the best/fastest segmenter. CPU is a correctness fallback; MPS/runtime support and throughput must be profiled |
| Mask cadence/policy | Mask every processed image of each selected camera; exclude people and recognized potentially mobile vehicles/animals, including parked vehicles | Start with current-frame masks rather than unvalidated propagation. Working confidence/binary-mask thresholds 0.5 and a 3-pixel boundary guard band are adjustable validation defaults |
| Pose window | Ten retained rig-pose clones | Approximately two seconds at 5 Hz; one clone per timestamp, not per camera. Adjust for parallax, memory and actual processing delay |
| Backend work | Start with at most 200 accepted feature tracks per rig update, selected for geometry/spatial coverage/camera diversity | Separate matching budget from filter-update budget. End or marginalize tracks without reusing consumed information |
| Development cache | Cache frozen masks, keypoints/descriptors and pairwise match candidates | Avoid repeated expensive frontend inference while training small estimator heads. Reference data must not select online features; state-dependent validation still uses the actual causal filter |

These are project defaults for investigation, not literature-established optimal values or measured real-time capability. Profile the complete frontend and backend and report image resolution, selected cameras, accepted tracks, delay and failures.

## Camera geometry and timing details

NCLT's five surrounding cameras and one upward camera are not a conventional wide-baseline stereo pair. The audited neighboring side-camera baselines are about 49 mm, with limited overlap. Complementary views and temporal tracking are therefore important even when simultaneous depth is weak. Choose exact IDs during the implementation image/calibration audit rather than guessing which file number faces forward. [Audited camera geometry](./nclt-sensors-and-state.md#clarification-ladybug-stereo-and-multiple-views).

### Why the farthest cameras are not automatically best

Recomputing center distances from the corrected calibration gives neighboring side pairs of **48.98–49.47 mm** and non-neighboring side pairs of **79.52–79.78 mm**. The largest side baseline is cam1–cam4 at 79.78 mm. These are calculations from calibration translations, not measured scene overlap or depth accuracy. [Corrected calibration archive](https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ladybug3_calib/cam_params.zip).

Ladybug views are distributed around the rig. The farther side pairs look in substantially different directions; even neighboring views have small overlap. A greater center distance helps depth only when the same static feature is observed with favorable ray geometry and adequate pixel/calibration accuracy. Occlusion, matching reliability and usable overlap can outweigh the larger baseline. [Schneider and Förstner, Section 4.3.2](https://www.dgpf.de/pfg/2013/pfg2013_4_schneider.pdf).

Two suitable calibrated views can triangulate full 3D feature coordinates, including relative height; a vertical-baseline camera is not required just to recover a third coordinate. This is distinct from absolute world-height observability. A third camera can improve coverage, triangulation conditioning or redundancy if its views contain useful common/static structure. An upward camera may provide mainly sky outdoors, so it is not automatically the best third camera.

Recommendation: inspect a small training-only sample and the calibration; rank candidate pairs by usable static tracks, overlap, triangulation angles/conditioning, spatial coverage and computation. Start with the best supported two-camera configuration. Consider a third if it provides measured information/robustness gains. The user has accepted the small initial camera count; the exact pair and third-camera benefit remain unverified.

Support the agreed one-to-six-camera configurations and 15 simultaneous pairs in six-camera all-pairs mode. The initial reduced graph attempts useful overlap edges only; each selected camera keeps its own temporal tracks. Fuse shared landmark/track observations consistently, not as independent duplicate poses. Same-timestamp stereo alone does not observe ego-motion without temporal or other world information. Retain tracks across distinct rig poses with sufficient geometry before accepting motion constraints.

Use physical acquisition times and supported synchronization tolerances; do not call arbitrarily offset frames simultaneous. Delayed observations attach to their retained acquisition pose. Measurements older than supported history require a declared delay policy rather than being attached to the current pose. Actual delay may require revising the window or processing schedule.

## Resolution and masking details

SuperPoint uses grayscale content; Mask R-CNN uses color. Preserve a consistent spatial mapping through orientation, distortion handling, resizing and mask projection. Extract features from original visual content and remove suspect keypoints; do not create synthetic mask edges by blacking out objects. Adjacent cameras do not require a fabricated conventional rectified-stereo model: use the calibrated multi-camera projection/bearing geometry.

The official LightGlue extractor auto-resizes unless disabled. Explicitly control that transform so an 800-pixel operating choice is not silently changed. Its pretrained SuperPoint matcher supports configurable feature counts and adaptive depth/width; begin with reference defaults and profile actual hardware. [Official LightGlue implementation/configuration](https://github.com/cvg/LightGlue).

Torchvision provides the proposed pretrained Mask R-CNN checkpoint and outputs instance masks/labels/scores. Control its internal resizing separately and map returned masks back to the tracking image correctly; the feature-image size does not by itself cap detector computation. Model-internal resize settings and package versions are implementation details to record. [Official model/checkpoint documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html).

Masking remains enabled in the intended baseline and can be disabled for ablations. Unknown movers, reflections and missed masks still require geometry/inertial consistency. Insufficient valid tracks mean a degraded/skipped visual update, not restored dynamic foreground. Reusing/warping masks at a reduced cadence is a later optimization requiring age/coverage checks, not the initial default.

## What this decision leaves to implementation

Choose actual camera IDs after inspecting a small image sample and corrected calibration; verify coordinate transforms, temporal/cross-view matches and triangulation. Check pretrained runtime/operator compatibility and cache storage requirements before scaling downloads. Adjust the proposed operating point using validation/compute measurements. A lower camera count or cached development replay does not demonstrate online end-to-end latency.
