# Temporal skeleton transformer model card

## Intended use

Sequence classification from tracked, normalized COCO-17 keypoints. The shipped real-data configuration
(`config/demo-real.yaml`, `config/benchmark-real.yaml`) uses fall-detection labels **`upright`,
`falling`, `fallen`**, framed as industrial worker-safety incident detection. The label set is
configurable; the synthetic demo keeps placeholder labels and a deterministic kinematic fallback.

## Architecture

Each frame is a 51-value vector (17 × x/y/confidence). A linear projection feeds three pre-norm
transformer encoder blocks (4 heads, 128 hidden dims) with a padding mask, masked mean temporal pooling,
and a linear head. Short windows are right-padded and the padding is masked out of attention and pooling.

## Training data and method (real)

- **Dataset:** UR Fall Detection Dataset (URFD), M. Kepski & B. Kwolek, University of Rzeszów —
  http://fenix.ur.edu.pl/~mkepski/ds/uf.html. Research use; downloaded locally, not redistributed here.
- YOLO26-pose extracts COCO-17 skeletons from cam0 RGB frames; 32-frame windows are labelled from the
  URFD per-frame annotation (`-1` upright, `0` falling, `1` fallen) by their last frame.
- **Split by sequence** (no window leakage); inverse-frequency class weights for imbalance.
- Pipeline: `scripts/prepare_urfd_dataset.py` → `scripts/train_temporal.py --val-dataset ...`.

## Evaluation (held-out sequences)

- Validation **macro-F1 = 0.90**, accuracy 0.93 on held-out fall sequences.
- Confusion matrix (rows = true `[upright, falling, fallen]`): `[[12,0,0],[3,26,0],[0,3,16]]`.
- The checkpoint stores label order, sequence length, seed, macro-F1, and accuracy. It is regenerated
  with `make dataset-real` and not committed (the dataset is not redistributed).

## Recommended reporting before site deployment

- Re-split by subject, session, and camera; report macro-F1, per-class precision/recall, calibration
  error, and confusion matrix on a site set.
- Evaluate missing/low-confidence joints, partial occlusion, camera motion, lighting, and ID switches.
- Tune `event_threshold` on a held-out operational set, not the training set.

## Limitations

URFD is a single-subject, staged indoor dataset; it demonstrates a real, honestly-evaluated fall
classifier but is not a substitute for site-specific labelled data. Behaviour labels are not clinical
diagnoses, and action classification must not directly control safety-critical equipment — keep a human
escalation path. The synthetic demo uses deterministic kinematic rules and declares that backend in
readiness; random untrained neural predictions are intentionally prohibited.
