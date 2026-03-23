"""
Wildlife Camera System - Raspberry Pi
======================================
Hardware connections:
  - 5V pin on raspberry pi -> VCC pin of PIR Motion sensor
  - GPIO 23 (raspberry pi) -> output PIR Motion Sensor 
  - GND (raspberry pi) -> GND pin of motion sensor
  - CSI connector camera -> NoIR Camera Module - used during day/night.


Logic:
1. Wait for motion from PIR sensor on GPIO23 
2. When motion is detected, trigger the Pi Camera Module 3 NoIR
3. Record/stream video for up to 30 seconds. 
4. Send video frames directly to EcoLens website. 
5. Stop recording after 30 seconds. 
"""

import cv2
import time
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
    print("[WARN] RPi.GPIO not found - running in SIMULATION mode")
    ON_PI = False

# - Pi Camera 
try: 
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError: 
    print("WARNING! Picamera2 not found - falling back to OpenCV camera mode.")
    HAS_PICAMERA2 = False
    

# ── Config ─────────────────────────────────────────────────────────────────
GPIO_MOTION      = 23      
RECORD_SECONDS = 30 # max recording duration per motion event
MOTION_COOLDOWN = 5 # time between captures 
FRAME_WIDTH = 1280 # camera frame width in pixels
FRAME_HEIGHT = 720 
FRAME_RATE = 20 


STREAM_ENDPOINT = "https://cmpe246.8745.cloudns.cl/cam/" # backend endpoint for streaming vids 


OUTPUT_DIR       = Path("detections")
MODEL_DIR        = Path("model")
LOG_FILE         = "wildlife_log.json"

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



#  GPIO / Hardware helpers

def setup_gpio(): #set up PIR motion sensor pin
    if not ON_PI: 
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
        # Simulation: trigger once every ~10 s for testing
        return (int(time.time()) % 10) == 0
    return bool(GPIO.input(GPIO_MOTION))



#  Camera helpers

def open_camera(index: int) -> cv2.VideoCapture:
    """ open raspberry pi camera module 3 NoIR."""
    """ Uses Picamera2 if available, otherwise falls back to OpenCV's Vidcapture"""

    if HAS_PICAMERA2: 
        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main = {"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}

        )
        picam2.configure(config)
        picam2.start()
        time.sleep(1.0)
        log.info("Pi Camera Module opened with Picamera2")
        return picam2, "picamera2"
    
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
    return cap, "opencv"

def close_camera(camera, mode: str): 
    """Close camera function"""
    if mode == "picamera2": 
        camera.stop()
    else: 
        camera.release()

def read_frame(camera, mode: str) -> np.ndarray: 
    # read one from (take a photo) from active camera 
    # Picamera2 returns RGB frames, so convert them to BGR to keep the rest of openCV code working as-is 

    if mode == "picamera2": 
        frame = camera.capture_array()
        if frame is None: 
            raise RuntimeError("Failed to capture frame from Picamera2")
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ret, frame = camera .read() 
    if not ret or frame is None: 
        raise RuntimeError("Failed to capture frame from OpenCV camera")
    return frame 



# ══════════════════════════════════════════════════════════════════════════════
#  Species detection (MobileNetV2 via OpenCV DNN)


def load_classifier():
    """
    Load MobileNetV2 (ImageNet) as an OpenCV DNN.
    Requires model files to already exist in the model/folder 
    Run download_model.py first 
    """
    proto = MODEL_DIR / "mobilenet_v2_deploy.prototxt"
    weights = MODEL_DIR / "mobilenet_v2.caffemodel"
    MODEL_DIR.mkdir(exist_ok=True)

    # If caffemodel not present, fall back to TensorFlow SavedModel approach
    # (OpenCV supports both; adjust as needed for your Pi setup)
    if not weights.exists() or not proto.exists():
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


def annotate_frame(frame: np.ndarray, detections: list[dict], timestamp: str) -> np.ndarray:
    """Draw detection labels and metadata onto the frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Overlay bar
    cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 0), -1)

    cv2.putText(annotated, f"{timestamp}  |  LIVE | NoIR camera",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y = 95
    if detections:
        top = detections[0]
        label = f"{top['label'].upper()}  {top['confidence'] * 100:.1f}%"
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


# ══════════════════════════════════════════════════════════════════════════════
#  Logging / persistence

def save_detection(detections: list[dict], timestamp: str):
    """Save detection info to JSON log."""
    record = {
        "timestamp": timestamp,
        "detections": detections,
    }

    log_path = OUTPUT_DIR / LOG_FILE
    existing = []

    if log_path.exists():
        with open(log_path) as f:
            existing = json.load(f)

    existing.append(record)

    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)

    log.info(f"Detection saved → {detections}")



# ══════════════════════════════════════════════════════════════════════════════
#  Main loop

import requests

def stream_frame(frame, timestamp):
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


def run():
    OUTPUT_DIR.mkdir(exist_ok=True)
    setup_gpio()

    net = load_classifier()
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

            log.info("Motion detected! Starting 30s recording...")

            try:
                camera, mode = open_camera(0)
            except RuntimeError as e:
                log.error(f"Camera error: {e}")
                continue
            detections = []
            start_time = time.time()

            while (time.time() - start_time) < RECORD_SECONDS:
                try:
                     frame = read_frame(camera, mode)

                except Exception as e:
                    log.error(f"Frame capture error: {e}")
                    break
               
                # classify
                detections = classify_frame(net, frame)

                # annotate
                annotated = annotate_frame(frame, detections, timestamp)

                # stream
                stream_frame(annotated, timestamp)

                # show preview (optional)
                cv2.imshow("Wildlife Camera", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            close_camera(camera, mode)

            # save detection (just last one is fine)
            save_detection(detections, timestamp)

            log.info("Recording complete")

    except KeyboardInterrupt:
        log.info("Shutting down")

    finally:
        cv2.destroyAllWindows()
        cleanup_gpio()
    
    log.info("System stopped.")

if __name__ == "__main__": 
    run() 




