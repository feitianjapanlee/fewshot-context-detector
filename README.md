# Few-shot Novel Object Detection Prototype

Context-image-conditioned object detection prototype.

## What this is
A validation pipeline for detecting object instances in a query image given:
- one or more classes
- one or more reference images per class
- no task-specific training on those classes

The implementation uses:
- GroundingDINO for generic region proposals
- CLIP for visual matching between proposals and reference images
- optional text scoring from class names

This is a practical prototype for evaluation and iteration, not a new foundation model.

## Why this design
Directly training a true generalized few-shot detector is expensive. For a usable baseline that works on novel classes without per-class training, a strong pattern is:
1. generate candidate boxes with an open-vocabulary detector
2. encode candidates and reference images into a shared embedding space
3. classify boxes by nearest reference-class prototype
4. apply NMS and confidence thresholds

This gives a zero additional training baseline and is easy to validate.

## Project layout
- `requirements.txt` Python dependencies
- `context_detector.py` main pipeline
- `demo_context.json` sample input format
- `run_demo.py` CLI entrypoint
- `evaluate.py` simple evaluator for prediction JSON against GT JSON

## Input format
```json
{
  "context": [
    {
      "class": "Tesla Model X",
      "refer_image": ["00001/front-view.jpg", "00001/rear-view.jpg"]
    },
    {
      "class": "Tesla Model Y",
      "refer_image": ["00002/front-view.jpg", "00002/rear-view.jpg"]
    }
  ]
}
```

## Output format
```json
{
  "image": "cars-on-road.jpg",
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "class": "Tesla Model X",
      "score": 0.83,
      "similarity": 0.79,
      "proposal_score": 0.87
    }
  ]
}
```

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python run_demo.py \
  --context demo_context.json \
  --query path/to/cars-on-road.jpg \
  --output outputs/predictions.json \
  --vis outputs/vis.jpg
```

## Notes
- GroundingDINO still needs a prompt to propose objects. This prototype uses a generic prompt like `object . vehicle . product . item .` to maximize recall.
- For some domains, proposal recall may be the limiting factor.
- If you want a stronger system later, likely upgrades are OWLv2 / GroundingDINO 1.5 / SAM2-assisted regioning / SigLIP2 embeddings.
