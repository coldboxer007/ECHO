# Agent1 Emotion Analysis Findings

> **Agent:** OpenCode / Agent 1  
> **Date:** 2026-04-08  
> **Scope:** Compare `emotion_test_perfect.py` against `camera_test.py` and re-evaluate the earlier emotion-analysis findings  
> **Environment Note:** Verification was done from the Mac workspace. Direct interactive visual validation was limited by the CLI environment, but model-level and startup-level checks were run locally.

---

## Executive Summary

After comparing the workflows directly, my earlier finding needs to be corrected in an important way.

The stale-state issue in `camera_sentiment.py` is real and still matters for the full robot runtime, but it is **not the main explanation** for why the standalone "perfect" script behaves better than `camera_test.py`.

The more important finding is that `emotion_test_perfect.py` and `camera_test.py` are **not running the same inference pipeline**.

`emotion_test_perfect.py` differs in three critical ways:

1. it uses a different label order
2. it uses different image preprocessing
3. it does not apply a second softmax when the model output is already normalized

Those differences are large enough to change the predicted emotion, including turning a `Neutral` top class into `Sad` or `Happy` depending on how the output is interpreted.

So, after this comparison, my updated conclusion is:

- `camera_test.py` and `camera_sentiment.py` are likely misinterpreting the model output
- `camera_sentiment.py` also has a stale-state bug that can make a wrong result persist longer
- camera quality may still affect confidence, but it is not the best primary explanation

---

## What Was Compared

- `emotion_test_perfect.py` - known-good standalone script
- `camera_test.py` - current standalone test utility that you reported as problematic
- `camera_sentiment.py` - runtime code used by `main.py` and `live_main.py`
- `config.py` - shared labels and threshold
- `fer_3stage_fp16.tflite` - actual model metadata and numeric behavior

---

## Verification Performed

### 1. `emotion_test_perfect.py` startup on this Mac

Run:

```bash
python3 emotion_test_perfect.py
```

Observed result:

```text
ai-edge-litert not installed. Run: pip install ai-edge-litert
```

This script could not be run end-to-end here because it hard-requires `ai-edge-litert` and exits immediately if it is missing.

### 2. Direct TFLite model inspection on this Mac

I inspected the actual model using `tensorflow.lite.Interpreter`.

Observed model metadata:

- input shape: `1 x 224 x 224 x 3`
- input dtype: `float32`
- output shape: `1 x 7`
- output dtype: `float32`

I also verified that the model output already sums to approximately `1.0`, which means it already behaves like a probability distribution.

### 3. `camera_test.py` startup smoke test

Run:

```bash
python3 camera_test.py --window
```

Observed result:

- process started successfully
- command exceeded the short timeout because it opens an interactive camera window and waits

So there is no evidence here that `camera_test.py` fails to start on this Mac.

### 4. Numeric comparison of the two inference pipelines

I compared the `emotion_test_perfect.py` pipeline and the `camera_test.py` pipeline against the same model.

Key observed result:

```text
model_outputs_already_probabilities? True

PERFECT PIPELINE
raw= [0.0774, 0.1338, 0.0379, 0.22, 0.2838, 0.1056, 0.1417]
sum= 1.0 argmax_idx= 4 label= Neutral conf= 0.2838
if_softmax_applied_again= sad 0.164

CAMERA_TEST PIPELINE
raw= [0.1102, 0.0586, 0.0526, 0.2748, 0.2474, 0.1348, 0.1216]
sum= 1.0 argmax_idx= 3 perfect_label= Happy camera_label= happy conf= 0.2748
after_camera_softmax= happy 0.1625
```

That is enough to show that the current `camera_test.py` logic is not equivalent to the working script.

---

## Main Findings

### 1. The label order in `camera_test.py` / `camera_sentiment.py` does not match `emotion_test_perfect.py`

**Relevant files:**

- `emotion_test_perfect.py:30`
- `config.py:72`

`emotion_test_perfect.py` uses:

```python
['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
```

`config.py` uses:

```python
["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
```

The first four entries match by position, but the last three do not:

- index 4: `Neutral` vs `sad`
- index 5: `Sad` vs `surprise`
- index 6: `Surprise` vs `neutral`

This is not a cosmetic difference. It changes the meaning of the model output.

A prediction that the perfect script would label as `Neutral` can be mislabeled by `camera_test.py` as `sad`.

This is a much stronger root-cause candidate than my earlier assumption that the pipelines were mostly the same.

### 2. The preprocessing pipeline is different

**Relevant files:**

- `emotion_test_perfect.py:68-71`
- `camera_test.py:121-150`
- `camera_sentiment.py:207-236`

`emotion_test_perfect.py` preprocessing:

- resize to `224x224`
- convert `BGR -> RGB`
- cast to `float32`
- keep values in raw `0..255`

`camera_test.py` preprocessing:

- resize to model size
- keep `BGR`
- divide by `255.0`
- cast to model dtype

That means the two scripts feed materially different tensors into the same model.

This is likely significant because many image models are trained with a very specific channel order and input range. The "perfect" script is explicitly hard-coded for this model, while `camera_test.py` is trying to generalize based only on tensor shape.

### 3. `camera_test.py` and `camera_sentiment.py` apply softmax even though the model output is already probabilities

**Relevant files:**

- `emotion_test_perfect.py:76-80`
- `camera_test.py:145-149`
- `camera_sentiment.py:231-236`

`emotion_test_perfect.py` does this carefully:

```python
if raw.min() < 0 or abs(raw.sum() - 1.0) > 0.01:
    e = np.exp(raw - raw.max())
    raw = e / e.sum()
```

So it only applies softmax if the output looks like logits.

`camera_test.py` and `camera_sentiment.py` always apply softmax.

That is incorrect for this specific model, because I verified numerically that the outputs already sum to `1.0`.

Applying softmax twice compresses the distribution and changes the meaning of the confidence values. It also makes thresholding less reliable.

Example from the verification:

- perfect pipeline raw top label: `Neutral` at `0.2838`
- if softmax is applied again, the top label becomes interpreted through a distorted distribution, with the top confidence collapsing to around `0.164`

So this is a real bug, not just a style preference.

### 4. The stale-state bug in `camera_sentiment.py` still exists, but it is secondary to the workflow mismatch

**Relevant file:** `camera_sentiment.py:181-274`

My earlier finding here still stands:

- early returns for no-face / no-interpreter / inference error do not clear shared emotion state
- `_current_emotion` can remain on the last successful prediction
- the main robot runtime can therefore appear stuck on a previous emotion

That means this bug is still worth fixing.

But after comparing the workflows, it should no longer be treated as the first thing to change in the standalone test path. The inference interpretation issues above are more foundational.

### 5. Camera quality still does not fully explain the problem

If camera quality were the main issue, I would still expect mostly:

- missed faces
- low confidence
- unstable predictions

I would not expect systematic differences caused by:

- a different label order
- different channel ordering and value scaling
- double-softmax on already-normalized outputs

Those are software issues, not camera issues.

So the updated conclusion remains: camera quality may contribute, but it is not the primary explanation.

---

## Workflow Comparison Summary

### `emotion_test_perfect.py`

1. load model with `ai-edge-litert`
2. detect face
3. resize face to `224x224`
4. convert `BGR -> RGB`
5. keep `float32` values in `0..255`
6. infer
7. only apply softmax if output does not already look normalized
8. map output using label order:
   `Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise`

### `camera_test.py`

1. load model with generic runtime fallback
2. detect face
3. resize face to model size
4. keep `BGR`
5. normalize to `0..1`
6. infer
7. always apply softmax
8. map output using label order from `config.py`:
   `angry, disgust, fear, happy, sad, surprise, neutral`

These are not the same workflow.

---

## Updated Recommendation Order

### Priority 1: Make `camera_test.py` match the working script exactly

Before tuning anything else, I would make `camera_test.py` use the same inference contract as `emotion_test_perfect.py`:

- same label order
- same preprocessing
- same conditional-softmax logic

This gives you the cleanest A/B check because `emotion_test_perfect.py` is your known-good reference.

### Priority 2: Apply the same fixes to `camera_sentiment.py`

Once `camera_test.py` matches the working script, make `camera_sentiment.py` use the same model contract too.

Specifically:

- update `SENTIMENT_LABELS` to the verified model order
- switch preprocessing to `BGR -> RGB`, `float32`, raw `0..255` if that is indeed what the model expects
- only apply softmax when output is not already normalized

### Priority 3: Then fix the stale-state bug in `camera_sentiment.py`

After the model interpretation is corrected, fix the runtime state bug:

- clear shared state on no-face / error paths
- decay EMA when no face is present
- avoid long-lived stale emotion values

This is still necessary for the robot runtime.

### Priority 4: Only after that, investigate camera quality and capture concurrency

If problems remain after the model contract is corrected, then investigate:

- low-quality Raspberry Pi frames
- lighting
- repeated `VideoCapture.read()` across threads
- face crop quality

That is the right stage to evaluate hardware limitations.

---

## Recommended Minimal Code Changes

### A. Fix label order in `config.py`

Current:

```python
SENTIMENT_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
```

Recommended first-pass alignment with the working script:

```python
SENTIMENT_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
```

### B. Fix preprocessing in `camera_test.py` and `camera_sentiment.py`

Recommended direction:

```python
face_input = cv2.resize(face_roi, (target_w, target_h))
face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB).astype(np.float32)
face_input = np.expand_dims(face_input, axis=0)
```

Do not divide by `255.0` unless you verify from model documentation that normalization is expected.

### C. Only apply softmax when needed

Recommended direction:

```python
raw = output[0].astype(np.float64)
if raw.min() < 0 or abs(raw.sum() - 1.0) > 0.01:
    shifted = raw - np.max(raw)
    exp_vals = np.exp(shifted)
    probabilities = exp_vals / np.sum(exp_vals)
else:
    probabilities = raw
```

### D. Fix stale state in `camera_sentiment.py`

Recommended direction:

```python
if len(faces) == 0:
    for label in SENTIMENT_LABELS:
        self._emotion_scores[label] *= 0.95
    with self._lock:
        self._current_emotion = "neutral"
        self._current_confidence = 0.0
    return "neutral", 0.0
```

---

## Best Validation Sequence

1. Make `camera_test.py` match `emotion_test_perfect.py` exactly.
2. Run both on the same Mac camera and compare live behavior.
3. Once they agree, port the same inference logic into `camera_sentiment.py`.
4. Then verify that the robot runtime returns to neutral when no face is present.
5. Only then decide whether Raspberry Pi camera quality is still a meaningful limiting factor.

---

## Final Conclusion

My earlier report was partially correct but incomplete.

Updated conclusion:

- the stale-state bug in `camera_sentiment.py` is real
- but the larger issue is that `camera_test.py` and `camera_sentiment.py` do not match the known-good workflow in `emotion_test_perfect.py`
- the strongest software problems are label-order mismatch, preprocessing mismatch, and unconditional double-softmax
- therefore the hypothesis that the Raspberry Pi camera is the main reason is **not supported** by the current evidence

The recommended next move is to make the failing pipelines match the working pipeline first, then re-test on Mac, then validate on Raspberry Pi.
