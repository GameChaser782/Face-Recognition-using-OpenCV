# Face-Recognition — Plan A (deferred)

This repo is the original VisionID college project (OpenCV + Streamlit + KNN-on-pixels
face recognition + Caffe age/gender). Renamed because the `VisionID` name has been
reused for a VLM document-understanding project.

## Planned upgrade (when revisited)

Replace the KNN-on-raw-pixels recognizer with modern learned embeddings + liveness:

1. **Embeddings**: ArcFace (InsightFace) or FaceNet, fine-tuned/eval on LFW + CASIA-WebFace.
2. **Detector**: swap Haar cascade for YOLOv8-face or SCRFD.
3. **Liveness / anti-spoof**: train a binary classifier on CelebA-Spoof; require pass before identity match.
4. **Eval**: TAR @ FAR=1e-3 on LFW; spoof-attack APCER/BPCER.
5. **Export**: ONNX, INT8 quantized, < 50 MB total.
6. **Demo**: ONNX Runtime Web — fully client-side webcam recognition on github.io.

## Notebooks to add

- `notebooks/01_arcface_finetune.ipynb`
- `notebooks/02_liveness_classifier.ipynb`
- `notebooks/03_onnx_export_quantize.ipynb`

## Out of scope

VLM/multimodal — that lives in the new `VisionID` repo.
