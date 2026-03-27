"""
Wildlife Camera System - Raspberry Pi
====================================

Hardware connections:
  - 5V pin on raspberry pi -> VCC pin of PIR Motion sensor
  - GPIO 23 (raspberry pi) -> output PIR Motion Sensor
  - GND (raspberry pi) -> GND pin of motion sensor
  - CSI connector camera -> NoIR Camera Module

Logic:
1. Wait for motion from PIR sensor on GPIO23
2. When motion is detected, trigger the camera stream
3. Record/stream video for up to 30 seconds
4. Run animal classification on frames
5. Save the most consistent detection for that motion event
"""

import cv2
import time
import json
import logging
import numpy as np
import requests

from datetime import datetime
from pathlib import Path
from collections import Counter


# ── GPIO (only available on real Pi hardware) ──────────────────────────────
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    print("[WARN] RPi.GPIO not found - running in SIMULATION mode")
    ON_PI = False


# ── Pi Camera availability check ────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    print("[WARN] Picamera2 not found - falling back to OpenCV/RTSP mode.")
    HAS_PICAMERA2 = False


# ── Config ──────────────────────────────────────────────────────────────────
GPIO_MOTION = 23
RECORD_SECONDS = 30
MOTION_COOLDOWN = 5
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 20

STREAM_ENDPOINT = "https://cmpe246.8745.cloudns.cl/cam/"

OUTPUT_DIR = Path("detections")
MODEL_DIR = Path("model")
LOG_FILE = "wildlife_log.json"

CONFIDENCE_MIN = 0.50

# Set to True if you want guaranteed fallback detections when model is absent
SIMULATE_DETECTIONS_IF_NO_MODEL = True

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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("camera_system.log")
    ]
)
log = logging.getLogger(__name__)


# ── GPIO helpers ────────────────────────────────────────────────────────────
def setup_gpio():
    if not ON_PI:
        log.info("GPIO simulation mode active")
        return

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_MOTION, GPIO.IN)
    log.info(f"GPIO initialized (BCM mode), motion sensor on GPIO{GPIO_MOTION}")


def cleanup_gpio():
    if ON_PI:
        GPIO.cleanup()


def read_motion() -> bool:
    """Return True when PIR detects motion."""
    if not ON_PI:
        # Simulation: trigger once every ~10 seconds
        return (int(time.time()) % 10) == 0
    return bool(GPIO.input(GPIO_MOTION))


# ── Camera helpers ──────────────────────────────────────────────────────────
def open_camera(source: str = "rtsp://127.0.0.1:8554/cam"):
    """
    Open RTSP stream using OpenCV.
    """
    log.info(f"Opening RTSP stream: {source}")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot connect to RTSP stream at {source}")

    return cap, "rtsp"


def close_camera(camera, mode: str):
    if mode == "picamera2":
        camera.stop()
    else:
        camera.release()


def read_frame(camera, mode: str) -> np.ndarray:
    if mode == "picamera2":
        frame = camera.capture_array()
        if frame is None:
            raise RuntimeError("Failed to capture frame from Picamera2")
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    ret, frame = camera.read()
    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame from OpenCV camera")
    return frame


# ── Model / classification ──────────────────────────────────────────────────
def load_classifier():
    """
    Load MobileNetV2 as an OpenCV DNN.
    If files are missing or loading fails, return None so the system can keep running.
    """
    proto = MODEL_DIR / "mobilenet_v2_deploy.prototxt"
    weights = MODEL_DIR / "mobilenet_v2.caffemodel"
    MODEL_DIR.mkdir(exist_ok=True)

    if not proto.exists() or not weights.exists():
        log.warning("Model files missing. Running without classifier model.")
        return None

    try:
        net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        log.info("MobileNetV2 classifier loaded")
        return net
    except Exception as e:
        log.error(f"Failed to load classifier: {e}")
        return None


def simulated_detection() -> list[dict]:
    """
    Fallback detections used when no model is available.
    """
    return [{
        "label": "brown bear",
        "confidence": 0.91,
        "class_id": 294
    }]


def classify_frame(net, frame: np.ndarray) -> list[dict]:
    """
    Return the best matching animal from ANIMAL_CLASSES for this frame.
    If net is None, optionally return a simulated brown bear detection.
    """
    if net is None:
        if SIMULATE_DETECTIONS_IF_NO_MODEL:
            return simulated_detection()
        return []

    try:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (224, 224)),
            scalefactor=1.0 / 127.5,
            size=(224, 224),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )

        net.setInput(blob)

        # Flatten to avoid ndarray-scalar conversion warnings
        predictions = net.forward().flatten()

        sorted_idx = np.argsort(predictions)[::-1]

        for idx in sorted_idx:
            idx = int(idx)
            conf = float(predictions[idx])

            if conf < CONFIDENCE_MIN:
                continue

            if idx in ANIMAL_CLASSES:
                return [{
                    "label": ANIMAL_CLASSES[idx],
                    "confidence": conf,
                    "class_id": idx
                }]

        return []

    except Exception as e:
        log.error(f"Classification error: {e}")
        return []


def annotate_frame(frame: np.ndarray, detections: list[dict], timestamp: str) -> np.ndarray:
    """Draw detection labels and metadata onto the frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 0), -1)

    cv2.putText(
        annotated,
        f"{timestamp} | LIVE | NoIR camera",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    y = 95
    if detections:
        top = detections[0]
        label = f"{top['label'].upper()} {top['confidence'] * 100:.1f}%"
        cv2.putText(
            annotated,
            label,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 80),
            2
        )
    else:
        cv2.putText(
            annotated,
            "No animal classification",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 140, 255),
            2
        )

    return annotated


# ── Persistence / logging ───────────────────────────────────────────────────
def save_detection(detections: list[dict], timestamp: str):
    """
    Save detection info to JSON log.
    """
    record = {
        "timestamp": timestamp,
        "detections": detections,
    }

    log_path = OUTPUT_DIR / LOG_FILE
    existing = []

    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                existing = json.load(f)
        except Exception:
            log.warning("Could not read existing wildlife log. Starting fresh.")
            existing = []

    existing.append(record)

    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)

    log.info(f"Detection saved -> {detections}")


def stream_frame(frame: np.ndarray, timestamp: str):
    """Send one frame to website."""
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        log.warning("Failed to encode frame for streaming")
        return

    try:
        requests.post(
            STREAM_ENDPOINT,
            files={"frame": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
            data={"timestamp": timestamp},
            timeout=1
        )
    except Exception as e:
        log.warning(f"Failed to stream frame: {e}")


def choose_best_detection(all_detections: list[dict]) -> list[dict]:
    """
    Choose the most consistent animal seen during the whole recording event.
    """
    if not all_detections:
        return []

    labels = [d["label"] for d in all_detections]
    most_common_label = Counter(labels).most_common(1)[0][0]

    matching = [d for d in all_detections if d["label"] == most_common_label]
    best = max(matching, key=lambda d: d["confidence"])

    return [best]


def log_special_detection(detections: list[dict]):
    """
    Explicitly print important terminal messages for certain animals.
    """
    if not detections:
        return

    label = detections[0]["label"].lower()

    if label == "brown bear":
        log.info("brown bear detected")
    else:
        log.info(f"{label} detected")


# ── Main loop ───────────────────────────────────────────────────────────────
def run():
    net = load_classifier()

    if net is None:
        log.warning("Classifier unavailable. Using simulation fallback if enabled.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    setup_gpio()

    last_capture = 0.0
    log.info("System started. Waiting for motion...")

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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            log.info(f"Motion detected! Starting {RECORD_SECONDS}s recording...")

            try:
                camera, mode = open_camera("rtsp://127.0.0.1:8554/cam")
            except RuntimeError as e:
                log.error(f"Camera error: {e}")
                continue

            all_detections = []
            start_time = time.time()

            while (time.time() - start_time) < RECORD_SECONDS:
                try:
                    frame = read_frame(camera, mode)
                except Exception as e:
                    log.error(f"Frame capture error: {e}")
                    break

                detections = classify_frame(net, frame)

                if detections:
                    all_detections.extend(detections)
                    log_special_detection(detections)

                annotated = annotate_frame(frame, detections, timestamp)
                stream_frame(annotated, timestamp)

                # Small delay helps avoid maxing CPU in tight loop
                time.sleep(0.03)

            close_camera(camera, mode)

            final_detections = choose_best_detection(all_detections)
            save_detection(final_detections, timestamp)

            if final_detections:
                log_special_detection(final_detections)
            else:
                log.info("No valid animal detected during this event")

            log.info("Recording complete")

    except KeyboardInterrupt:
        log.info("Shutting down")

    finally:
        cv2.destroyAllWindows()
        cleanup_gpio()
        log.info("System stopped.")


if __name__ == "__main__":
    run()


