# HF Space - DREEM Inference

## Status: Ready for Testing

### Completed
- [x] Create project structure
- [x] Create TODO.md
- [x] Create Streamlit app (app.py)
- [x] Create Dockerfile for HF Spaces
- [x] Create requirements.txt
- [x] Create README.md for Space (with HF Space metadata)
- [x] Add max_detection_overlap, confidence_threshold, max_tracks parameters
- [x] Add Streamlit config with upload size limit (200MB)
- [x] Add video validation (frame count check with warning/limit)
- [x] Implement interactive frame viewer with session state
- [x] Add track parsing from output .slp file
- [x] Add frame annotation overlay (keypoints, bounding boxes, labels)
- [x] Add navigation buttons (First/Prev/Next/Last Track)
- [x] Add track statistics panel
- [x] Add Lightning batch progress callback
- [x] Add pretrained model dropdown (animals/microscopy)
- [x] Auto-download models from HuggingFace Hub

### Pending
- [ ] Test locally with `docker build` and `docker run`
- [ ] Deploy to HF Spaces
- [ ] Add pre-trained model checkpoint (user needs to provide or host)

## Local Testing

```bash
# Build the Docker image
cd hf_space
docker build -t dreem-space .

# Run locally
docker run -p 7860:7860 dreem-space

# Access at http://localhost:7860
```

## Deploying to HF Spaces

1. Create a new Space on Hugging Face (https://huggingface.co/new-space)
2. Select "Docker" as the SDK
3. Upload or push these files:
   - `app.py`
   - `Dockerfile`
   - `requirements.txt`
   - `README.md`

Or use the HF CLI:
```bash
huggingface-cli login
huggingface-cli repo create dreem-tracking --type space
cd hf_space
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/dreem-tracking
git add .
git commit -m "Initial commit"
git push -u origin main
```
