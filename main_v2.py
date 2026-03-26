"""
Wildlife Camera System - Raspberry Pi 4B
==========================================
Hardware connections:
  - 5V pin  -> VCC pin of PIR Motion sensor
  - GPIO 23 -> OUT pin of PIR Motion sensor
  - GND     -> GND pin of PIR Motion sensor
  - CSI     -> Pi Camera Module 3 NoIR

Logic:
1. Wait for motion from PIR sensor on GPIO 23 (person walks past)
2. When motion is detected, open the Pi Camera Module 3 NoIR
3. Capture frames of the screen displaying a static animal image
4. Classify each frame with MobileNetV2 (ImageNet) via OpenCV DNN
5. Stream annotated frames to the EcoLens back-end
6. Save the best detection to a JSON log
"""

import cv2
import time
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import Counter

# ── GPIO ──────────────────────────────────────────────────────────────────────
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    print("[WARN] RPi.GPIO not found – running in SIMULATION mode")
    ON_PI = False

# ── Picamera2 ─────────────────────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    print("[WARN] Picamera2 not found – falling back to OpenCV /dev/video0")
    HAS_PICAMERA2 = False

# ── requests ──────────────────────────────────────────────────────────────────
try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    print("[WARN] 'requests' not installed – frame streaming disabled")
    HAS_REQUESTS = False

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

GPIO_MOTION     = 23
RECORD_SECONDS  = 30
MOTION_COOLDOWN = 5
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
FRAME_RATE      = 20

STREAM_ENDPOINT = "https://cmpe246.8745.cloudns.cl/cam/"

OUTPUT_DIR = Path("detections")
MODEL_DIR  = Path("model")
LOG_FILE   = "wildlife_log.json"

# Minimum confidence threshold (0–1). Lower = more detections but more
# false positives. 0.10 works well for clear screen images.
CONFIDENCE_MIN = 0.10

# ── ImageNet class indices for MobileNetV2 Caffe (shicai/MobileNet-Caffe) ────
# FIX: The original indices were wrong. These have been verified against the
# synset_words.txt file in the shicai/MobileNet-Caffe repository, which
# defines the exact output ordering for this specific model.
ANIMAL_CLASSES = {
     269: "timber wolf",
    270: "white wolf",
    271: "red wolf",
    272: "coyote",
    151: "Chihuahua", 


    277: "red fox",
    278: "kit fox",
    279: "arctic fox",
    280: "grey fox",

    281: "tabby",
    282: "tiger cat",
    283: "Persian cat",
    284: "Siamese cat",
    285: "Egyptian cat",

    286: "cougar",
    287: "lynx",
    290: "jaguar",
    291: "lion",
    292: "tiger",

    294: "brown bear",
    295: "black bear",
    296: "polar bear",
    297: "sloth bear",

    330: "wood rabbit",
    331: "hare",
    332: "Angora rabbit",
    333: "hamster",
    334: "porcupine",
    335: "fox squirrel",
    336: "marmot",
    337: "beaver",
    338: "guinea pig",

    339: "horse",
    340: "zebra",
    347: "bison",
    348: "ram",
    352: "deer",
    353: "deer",

    356: "weasel",
    357: "mink",
    358: "polecat",
    359: "black-footed ferret",
    360: "otter",
    361: "skunk",
    362: "badger",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("camera_system.log"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  GPIO helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_gpio():
    """Set up BCM GPIO with a pull-down resistor on the PIR input pin.

    FIX: The original code called GPIO.setup() without pull_up_down,
    leaving the pin floating. A floating pin reads random values and makes
    the PIR appear to fire constantly or never.
    HC-SR501 / HC-SR505 modules drive OUT HIGH on motion and return to
    near-zero otherwise, so PUD_DOWN gives a clean LOW baseline.
    """
    if not ON_PI:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_MOTION, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # ← KEY FIX
    log.info(f"GPIO ready – PIR on BCM {GPIO_MOTION} (pull-down enabled)")


def cleanup_gpio():
    if ON_PI:
        GPIO.cleanup()


_sim_last_trigger = 0.0  # simulation only

def read_motion() -> bool:
    """Return True when the PIR sensor detects motion.

    FIX (simulation mode): The original `(int(time.time()) % 10) == 0`
    expression is True for exactly one integer second per 10-second
    window. The main loop runs every 0.1 s, so there is only a ~1-in-100
    chance of sampling that second – it almost never fired. Replaced with
    a cooldown timer that reliably triggers every ~12 s for testing.
    """
    if not ON_PI:
        global _sim_last_trigger
        now = time.time()
        if (now - _sim_last_trigger) >= 12:
            _sim_last_trigger = now
            return True
        return False
    return bool(GPIO.input(GPIO_MOTION))


# ─────────────────────────────────────────────────────────────────────────────
#  Camera helpers
# ─────────────────────────────────────────────────────────────────────────────

def open_camera():
    """Open the Pi Camera Module 3 NoIR via Picamera2, or fall back to OpenCV.

    FIX: The original code always opened an RTSP stream and never actually
    used Picamera2 despite importing it. On a Pi 4B with Camera Module 3,
    Picamera2 is the correct and only supported driver.
    """
    if HAS_PICAMERA2:
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": FRAME_RATE},
        )
        cam.configure(config)
        cam.start()
        time.sleep(2)  # allow auto-exposure to settle before capturing
        log.info("Picamera2 opened (Camera Module 3 NoIR)")
        return cam, "picamera2"

    # Fallback for development / testing without a Pi camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FRAME_RATE)
    if not cap.isOpened():
        raise RuntimeError("Cannot open any camera (tried Picamera2 and /dev/video0)")
    log.info("OpenCV camera opened on /dev/video0")
    return cap, "opencv"


def close_camera(camera, mode: str):
    if mode == "picamera2":
        camera.stop()
    else:
        camera.release()


def read_frame(camera, mode: str) -> np.ndarray:
    """Capture one frame and return it as a BGR NumPy array."""
    if mode == "picamera2":
        frame = camera.capture_array()
        if frame is None:
            raise RuntimeError("Picamera2 returned None")
        # Picamera2 gives RGB; OpenCV expects BGR
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ret, frame = camera.read()
    if not ret or frame is None:
        raise RuntimeError("OpenCV camera read() failed")
    return frame


# ─────────────────────────────────────────────────────────────────────────────
#  Classifier  (MobileNetV2 via OpenCV DNN)
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier():
    """Load MobileNetV2 Caffe model. Run download_model.py first."""
    proto   = MODEL_DIR / "mobilenet_v2_deploy.prototxt"
    weights = MODEL_DIR / "mobilenet_v2.caffemodel"

    if not proto.exists() or not weights.exists():
        raise FileNotFoundError(
            f"Model files missing. Run  python3 download_model.py  first.\n"
            f"Expected:\n  {proto}\n  {weights}"
        )

    net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    log.info("MobileNetV2 classifier loaded")
    return net


def classify_frame(net, frame: np.ndarray) -> list[dict]:
    """Run MobileNetV2 inference and return the best matching animal.

    The preprocessing (scale=1/127.5, mean=127.5) matches the normalisation
    used when the shicai MobileNetV2 Caffe model was trained.
    """
    if net is None:
        return []

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (224, 224)),
        scalefactor=1.0 / 127.5,
        size=(224, 224),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    predictions = net.forward()[0]

    for idx in np.argsort(predictions)[::-1]:
        idx  = int(idx)
        conf = float(predictions[idx])
        if conf < CONFIDENCE_MIN:
            break  # all remaining scores are lower
        if idx in ANIMAL_CLASSES:
            return [{"label": ANIMAL_CLASSES[idx], "confidence": conf, "class_id": idx}]

    return []


# ─────────────────────────────────────────────────────────────────────────────
#  Frame annotation
# ─────────────────────────────────────────────────────────────────────────────

def annotate_frame(frame: np.ndarray, detections: list[dict], timestamp: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    cv2.rectangle(out, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(out, f"{timestamp}  |  LIVE | NoIR camera",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y = 95
    if detections:
        top   = detections[0]
        label = f"{top['label'].upper()}  {top['confidence'] * 100:.1f}%"
        cv2.putText(out, label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2)
    else:
        cv2.putText(out, "No animal detected",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 140, 255), 2)

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_detection(detections: list[dict], timestamp: str):
    record   = {"timestamp": timestamp, "detections": detections}
    log_path = OUTPUT_DIR / LOG_FILE
    existing = []
    if log_path.exists():
        with open(log_path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    existing.append(record)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
    log.info(f"Detection saved → {detections}")


# ─────────────────────────────────────────────────────────────────────────────
#  Streaming
# ─────────────────────────────────────────────────────────────────────────────

def stream_frame(frame: np.ndarray, timestamp: str):
    if not HAS_REQUESTS:
        return
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        log.warning("JPEG encoding failed – skipping stream")
        return
    try:
        _requests.post(
            STREAM_ENDPOINT,
            files={"frame": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            data={"timestamp": timestamp},
            timeout=1,
        )
    except Exception as exc:
        log.warning(f"Stream POST failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  Detection aggregation
# ─────────────────────────────────────────────────────────────────────────────

def choose_best_detection(all_detections: list[dict]) -> list[dict]:
    """Return the most-frequently-seen animal at its highest confidence."""
    if not all_detections:
        return []
    label = Counter(d["label"] for d in all_detections).most_common(1)[0][0]
    best  = max((d for d in all_detections if d["label"] == label),
                key=lambda d: d["confidence"])
    return [best]


# ─────────────────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run():
    try:
        net = load_classifier()
    except Exception as exc:
        log.error(f"Classifier load failed: {exc}")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    setup_gpio()

    last_capture = 0.0
    log.info("System ready – waiting for motion...")

    try:
        while True:
            if not read_motion():
                time.sleep(0.1)
                continue

            now = time.time()
            if (now - last_capture) < MOTION_COOLDOWN:
                time.sleep(0.1)
                continue

            last_capture = now
            timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
            log.info("Motion detected! Starting 30-second recording...")

            try:
                camera, mode = open_camera()
            except RuntimeError as exc:
                log.error(f"Camera error: {exc}")
                continue

            all_detections = []
            start_time     = time.time()

            while (time.time() - start_time) < RECORD_SECONDS:
                try:
                    frame = read_frame(camera, mode)
                except Exception as exc:
                    log.error(f"Frame capture error: {exc}")
                    break

                detections = classify_frame(net, frame)
                if detections:
                    all_detections.extend(detections)

                annotated = annotate_frame(frame, detections, timestamp)
                stream_frame(annotated, timestamp)

            close_camera(camera, mode)

            final = choose_best_detection(all_detections)
            save_detection(final, timestamp)
            log.info("Recording complete.")

    except KeyboardInterrupt:
        log.info("Shutting down (KeyboardInterrupt)")

    finally:
        cv2.destroyAllWindows()
        cleanup_gpio()

    log.info("System stopped.")


if __name__ == "__main__":
    run()