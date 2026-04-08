import numpy as np
import cv2

import camera_test


def _fake_details(shape=(1, 224, 224, 3), dtype=np.float32):
    return [{"shape": np.array(shape, dtype=np.int32), "dtype": dtype, "index": 0}]


class _FakeInterpreter:
    def __init__(self, output):
        self.output = np.array([output], dtype=np.float32)
        self.last_tensor = None

    def set_tensor(self, index, value):
        self.last_tensor = value

    def invoke(self):
        return None

    def get_tensor(self, index):
        return self.output


def test_camera_test_uses_working_model_label_order():
    assert camera_test.MODEL_LABELS == [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]


def test_run_inference_uses_rgb_raw_float32_input():
    face_roi = np.array(
        [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]],
        dtype=np.uint8,
    )
    fake = _FakeInterpreter([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
    input_details = _fake_details(shape=(1, 2, 2, 3), dtype=np.float32)
    output_details = [{"index": 0}]

    camera_test.run_inference(fake, input_details, output_details, face_roi)

    expected = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB).astype(np.float32)
    expected = np.expand_dims(expected, axis=0)
    np.testing.assert_allclose(fake.last_tensor, expected)


def test_run_inference_does_not_softmax_normalized_output_again():
    normalized = [0.05, 0.10, 0.07, 0.11, 0.31, 0.15, 0.21]
    fake = _FakeInterpreter(normalized)
    input_details = _fake_details()
    output_details = [{"index": 0}]
    face_roi = np.zeros((224, 224, 3), dtype=np.uint8)

    probs = camera_test.run_inference(fake, input_details, output_details, face_roi)

    np.testing.assert_allclose(probs, np.array(normalized, dtype=np.float64))
    assert int(np.argmax(probs)) == 4
    assert camera_test.MODEL_LABELS[int(np.argmax(probs))] == "neutral"
