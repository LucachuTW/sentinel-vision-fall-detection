# Local model artifacts

Model binaries are intentionally excluded from Git. The application never requires a cloud inference API.

Downloaded development artifacts can be verified with `sha256sum -c models/checksums.sha256` from the repository root.

| Artifact | Development default | Production recommendation |
|---|---|---|
| Pose | `models/yolo26n-pose.pt` | calibrated `models/yolo26s-pose-int8.engine` |
| Segmentation | `models/sam2.1_t.pt` | `models/sam2.1_t.pt` or a domain-fine-tuned SAM 2.1 checkpoint |
| Action | auditable kinematic fallback | `temporal-transformer.pt` trained on site-labelled skeleton windows |
| Narrative | disabled | Ollama `qwen3:1.7b`, isolated from safety decisions |

Ultralytics downloads supported PyTorch checkpoints on first use. For deterministic production deployments, download them during image build, verify their SHA-256 digest, and mount this directory read-only.

Do not deploy a randomly initialized temporal classifier. `scripts/train_temporal.py` refuses malformed or trivially small datasets and stores label order in the checkpoint; runtime startup rejects label mismatches.
