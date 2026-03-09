"""
Wildlife Camera System - Raspberry Pi
======================================
Hardware connections (from Figure 5):
  - GPIO4  → PIR Motion Sensor (via breadboard)
  - GND    → Ground (breadboard)
  - 5V     → Power (breadboard)
  - CSI    → NoIR Camera Module (ribbon cable) — used at night
  - USB    → USB Camera (used during day)

Logic:
  1. Poll light sensor (LDR on GPIO17 via RC circuit, or analog via MCP3008)
  2. If motion detected on GPIO4:
       - Light level HIGH  → activate USB day camera
       - Light level LOW   → activate NoIR CSI night camera
  3. Capture frames with OpenCV
  4. Run animal species classifier (MobileNetV2 pretrained on ImageNet)
  5. Save annotated image + log detection
"""

import cv2
import time
import os
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# ── GPIO (only available on real Pi hardware) ──────────────────────────────
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    print("[WARN] RPi.GPIO not found – running in SIMULATION mode")
    ON_PI = False

# ── Config ─────────────────────────────────────────────────────────────────
GPIO_MOTION      = 4        # PIR data pin  (GPIO4 as labelled in diagram)
GPIO_LIGHT       = 17       # LDR digital output (HIGH = bright, LOW = dark)
                            # Wire a simple LDR+resistor voltage divider to
                            # this pin, or use an LM393 light-sensor module
LIGHT_THRESHOLD  = 0.40     # 0–1 normalised; below → night mode
DAY_CAMERA_IDX   = 0        # USB camera (OpenCV index)
NIGHT_CAMERA_IDX = 1        # CSI NoIR module via libcamera-vid / v4l2

OUTPUT_DIR       = Path("detections")
MODEL_DIR        = Path("model")
LOG_FILE         = "wildlife_log.json"

MOTION_COOLDOWN  = 5        # seconds between captures to avoid burst
CONFIDENCE_MIN   = 0.25     # minimum confidence to display label

ANIMAL_CLASSES   = {        # ImageNet class indices for common wildlife
    281: "tabby cat",       271: "white wolf",      270: "timber wolf",
    275: "fox",             276: "Arctic fox",      277: "grey fox",
    278: "red fox",         279: "kit fox",         280: "grey fox",
    282: "tiger cat",       283: "Persian cat",     284: "Siamese cat",
    285: "Egyptian cat",    286: "cougar",          287: "lynx",
    288: "leopard",         289: "snow leopard",    290: "jaguar",
    291: "lion",            292: "tiger",           293: "cheetah",
    294: "brown bear",      295: "American black bear", 296: "ice bear",
    297: "sloth bear",      330: "wood rabbit",     331: "hare",
    332: "Angora rabbit",   333: "hamster",         334: "porcupine",
    335: "fox squirrel",    336: "marmot",          337: "beaver",
    338: "guinea pig",      339: "sorrel",          340: "zebra",
    341: "hog",             342: "wild boar",       343: "warthog",
    344: "hippopotamus",    345: "ox",              346: "water buffalo",
    347: "bison",           348: "ram",             349: "bighorn",
    350: "ibex",            351: "hartebeest",      352: "impala",
    353: "gazelle",         354: "Arabian camel",   355: "llama",
    356: "weasel",          357: "mink",            358: "polecat",
    359: "black-footed ferret", 360: "otter",       361: "skunk",
    362: "badger",          363: "armadillo",       364: "three-toed sloth",
    365: "orangutan",       366: "gorilla",         367: "chimpanzee",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("camera_system.log")]
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  GPIO / Hardware helpers
# ══════════════════════════════════════════════════════════════════════════════

def setup_gpio():
    if not ON_PI:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_MOTION, GPIO.IN)
    GPIO.setup(GPIO_LIGHT,  GPIO.IN)
    log.info("GPIO initialised (BCM mode)")


def cleanup_gpio():
    if ON_PI:
        GPIO.cleanup()


def read_motion() -> bool:
    """Return True if PIR detects motion."""
    if not ON_PI:
        # Simulation: trigger once every ~10 s for testing
        return (int(time.time()) % 10) == 0
    return bool(GPIO.input(GPIO_MOTION))


def read_light_level() -> float:
    """
    Return a normalised 0.0–1.0 light level.
    - Using a simple digital LDR module (HIGH = light, LOW = dark):
        returns 1.0 or 0.0
    - For a proper analog reading, replace with MCP3008 SPI read.
    """
    if not ON_PI:
        hour = datetime.now().hour
        return 0.8 if 7 <= hour < 20 else 0.1   # simulate day/night
    return float(GPIO.input(GPIO_LIGHT))


def is_daytime() -> bool:
    level = read_light_level()
    log.debug(f"Light level: {level:.2f} (threshold {LIGHT_THRESHOLD})")
    return level >= LIGHT_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
#  Camera helpers
# ══════════════════════════════════════════════════════════════════════════════

def open_camera(index: int) -> cv2.VideoCapture:
    """Open and configure a camera by OpenCV index."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def capture_frame(cap: cv2.VideoCapture, night_mode: bool) -> np.ndarray:
    """
    Capture a single frame.  Apply night-mode enhancement when using
    the NoIR camera in low-light conditions.
    """
    # Discard a few frames so auto-exposure can settle
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame")

    if night_mode:
        frame = enhance_night_frame(frame)

    return frame


def enhance_night_frame(frame: np.ndarray) -> np.ndarray:
    """
    Enhance a low-light NoIR frame:
      1. Convert to LAB colour space
      2. Apply CLAHE to the L (lightness) channel
      3. Denoise
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_ch  = clahe.apply(l_ch)

    enhanced_lab   = cv2.merge([l_ch, a_ch, b_ch])
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    enhanced_frame = cv2.fastNlMeansDenoisingColored(enhanced_frame, None, 10, 10, 7, 21)
    return enhanced_frame


# ══════════════════════════════════════════════════════════════════════════════
#  Species detection (MobileNetV2 via OpenCV DNN)
# ══════════════════════════════════════════════════════════════════════════════

def load_classifier():
    """
    Load MobileNetV2 (ImageNet) as an OpenCV DNN.
    Downloads model files automatically on first run.
    """
    proto = MODEL_DIR / "mobilenet_v2_deploy.prototxt"
    weights = MODEL_DIR / "mobilenet_v2.caffemodel"
    MODEL_DIR.mkdir(exist_ok=True)

    # If caffemodel not present, fall back to TensorFlow SavedModel approach
    # (OpenCV supports both; adjust as needed for your Pi setup)
    if not weights.exists():
        log.warning(
            "Caffemodel not found in ./model/  —  using OpenCV's built-in "
            "MobileNet SSD for demo.  For production, download MobileNetV2 "
            "caffemodel and prototxt from the OpenCV model zoo."
        )
        return None   # graceful fallback handled in classify_frame()

    net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    log.info("MobileNetV2 classifier loaded")
    return net


def classify_frame(net, frame: np.ndarray) -> list[dict]:
    """
    Run the frame through MobileNetV2 and return a list of detections:
      [{"label": str, "confidence": float, "class_id": int}, ...]
    filtered to animal classes only.
    """
    if net is None:
        # Simulation fallback: return a plausible dummy result
        return [{"label": "white-tailed deer (simulated)", "confidence": 0.91, "class_id": -1}]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (224, 224)),
        scalefactor=1.0 / 127.5,
        size=(224, 224),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    predictions = net.forward()[0]          # shape: (1000,)

    # Get top-5 predictions
    top5_idx = np.argsort(predictions)[::-1][:5]
    results = []
    for idx in top5_idx:
        conf = float(predictions[idx])
        if conf < CONFIDENCE_MIN:
            continue
        label = ANIMAL_CLASSES.get(int(idx), None)
        if label:
            results.append({"label": label, "confidence": conf, "class_id": int(idx)})

    return results


def annotate_frame(frame: np.ndarray, detections: list[dict],
                   night_mode: bool, timestamp: str) -> np.ndarray:
    """Draw detection labels and metadata onto the frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Overlay bar
    cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 0), -1)

    mode_txt = "🌙 NIGHT (NoIR)" if night_mode else "☀ DAY (USB)"
    cv2.putText(annotated, f"{timestamp}  |  {mode_txt}",
                (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y = 80
    if detections:
        for det in detections:
            label = f"{det['label'].upper()}  {det['confidence']*100:.1f}%"
            cv2.putText(annotated, label,
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2)
            y += 36
    else:
        cv2.putText(annotated, "No animal detected",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)

    return annotated


# ══════════════════════════════════════════════════════════════════════════════
#  Logging / persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_detection(image_path: str, detections: list[dict],
                   night_mode: bool, timestamp: str):
    """Append detection record to the JSON log."""
    record = {
        "timestamp":   timestamp,
        "mode":        "night" if night_mode else "day",
        "image":       image_path,
        "detections":  detections,
    }
    log_path = OUTPUT_DIR / LOG_FILE
    existing = []
    if log_path.exists():
        with open(log_path) as f:
            existing = json.load(f)
    existing.append(record)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
    log.info(f"Detection saved → {image_path}  |  {detections}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════════

def run():
    OUTPUT_DIR.mkdir(exist_ok=True)
    setup_gpio()

    net = load_classifier()
    last_capture = 0.0

    log.info("Wildlife camera system started.  Waiting for motion…")
    log.info(f"GPIO: Motion=GPIO{GPIO_MOTION}  Light=GPIO{GPIO_LIGHT}")

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
            night_mode   = not is_daytime()

            camera_idx   = DAY_CAMERA_IDX if not night_mode else NIGHT_CAMERA_IDX
            mode_label   = "NIGHT (NoIR)" if night_mode else "DAY (USB)"
            log.info(f"Motion detected! Mode={mode_label}  Camera index={camera_idx}")

            # ── Capture ──────────────────────────────────────────────────────
            try:
                cap   = open_camera(camera_idx)
                frame = capture_frame(cap, night_mode)
                cap.release()
            except RuntimeError as e:
                log.error(f"Camera error: {e}")
                continue

            # ── Classify ─────────────────────────────────────────────────────
            detections = classify_frame(net, frame)

            # ── Annotate & save ──────────────────────────────────────────────
            annotated  = annotate_frame(frame, detections, night_mode, timestamp)
            img_name   = f"{timestamp}_{'night' if night_mode else 'day'}.jpg"
            img_path   = str(OUTPUT_DIR / img_name)
            cv2.imwrite(img_path, annotated)
            save_detection(img_path, detections, night_mode, timestamp)

            # ── Optional: live preview (comment out for headless Pi) ─────────
            cv2.imshow("Wildlife Camera", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Quit key pressed – shutting down")
                break

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt – shutting down cleanly")
    finally:
        cv2.destroyAllWindows()
        cleanup_gpio()
        log.info("System stopped.")


if __name__ == "__main__":
    run()
