Download the InsightFace model pack (SCRFD detector + ArcFace recognizer) used by
Howdy's ONNX recognition backend.

```shell
# Downloads and unpacks buffalo_s into ./models/buffalo_s/
./install.sh

# Or pick the larger, slightly more robust pack
./install.sh buffalo_l
```

The packs come from the InsightFace release page:
https://github.com/deepinsight/insightface/releases/tag/v0.7

The recognition backend expects the `.onnx` files under `models/<pack>/`.
