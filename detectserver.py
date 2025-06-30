import cv2
import imageio
import numpy as np
import os
import requests
import logging
import paho.mqtt.client as mqtt
import json
import time
import shutil
import ssl

from ultralytics import YOLO
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

# Setup logging with timestamp and log rotation
log_file = os.getenv("LOG_FILE", "/tmp/detectionserver.log")
logger = logging.getLogger("DetectionServer")
logger.setLevel(logging.INFO)
handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
logger.addHandler(handler)

# MQTT Broker details
BROKER = os.getenv("MQTT_BROKER")
PORT = int(os.getenv("MQTT_PORT", 8883))
USERNAME = os.getenv("MQTT_USER")
PASSWORD = os.getenv("MQTT_PASS")
TOPIC_MOTION = os.getenv("TOPIC_MOTION")
TOPIC_CAT_ACTIVE = os.getenv("TOPIC_CAT")

# Get the cat details
cat_names_env = os.getenv("CAT_NAMES")
if not cat_names_env:
    raise RuntimeError("Environment variable CAT_NAMES is required but not set.")
CAT_NAMES = cat_names_env.split(",")

# GIF Generation Tuning
FRAME_INTERVAL = 0.5 #How often to take a capture from IMAGE_URL
DURATION = 5 #Length of the gif
OUTPUT_GIF = os.getenv("OUTPUT_GIF", "/tmp/catgif.gif")

# YOLO Model
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
logger.info("Loading YOLO model...")
model = YOLO(YOLO_MODEL_PATH)
logger.info("YOLO model loaded successfully.")
YOLO_CONFIDENCE = 0.7

# Image settings
SAVE_FOLDER = os.getenv("SAVE_FOLDER", "/tmp/cattrays")
IMAGE_URL = os.getenv("IMAGE_URL")
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/cattray_gif")

os.makedirs(SAVE_FOLDER, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Timing and state
MOTION_COOLDOWN = 180
last_motion_time = 0
active_cats = set()

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def publish_message(topic, payload):
    try:
        client.publish(topic, payload)
        logger.info(f"Published to {topic}: {payload}")
    except Exception as e:
        logger.error(f"Error publishing to {topic}: {e}")

def detect_cats_and_notify(image):
    global last_motion_time, active_cats
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"tapodetect-{timestamp}"
    unprocessed_image_path = os.path.join(SAVE_FOLDER, f"{base_name}.jpg")
    processed_image_path = os.path.join(SAVE_FOLDER, f"tapocats-processed-{timestamp}.jpg")

    try:
        cv2.imwrite(unprocessed_image_path, image)
        logger.info(f"Saved unprocessed image to: {unprocessed_image_path}")

        logger.info("Running YOLO inference...")
        results = model(image)
        detected_cats = set()

        for box in results[0].boxes:
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            label = model.names[class_id]
            if label in CAT_NAMES:
                if confidence > YOLO_CONFIDENCE:
                    detected_cats.add(label)
                    logger.info(f"Detected {label} with confidence {confidence:.2f}")
                else:
                    logger.info(f"Detected {label} with confidence {confidence:.2f}, too low to notify.")

        processed_image = results[0].plot()
        cv2.imwrite(processed_image_path, processed_image)
        logger.info(f"Saved processed image to: {processed_image_path}")

        if detected_cats:
            active_cats.update(detected_cats)
            message = f"{', '.join(detected_cats)} detected in the litter tray."
            logger.info(message)
            send_telegram_gif(message)
            last_motion_time = time.time()
        else:
            logger.info("No cat detected. Skipping Telegram notification.")
    except Exception as e:
        logger.error(f"Error during YOLO processing or file I/O: {e}")

def send_telegram_snapshot(caption, image_path): #If you don't want to send a GIF then use this to send an image instead
    logger.info("Sending snapshot to Telegram...")
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        image = cv2.imread(image_path)
        if image is not None:
            cropped = image[80:, :900]
            _, buffer = cv2.imencode(".jpg", cropped)
            photo_bytes = buffer.tobytes()
            files = {"photo": ("cropped.jpg", photo_bytes, "image/jpeg")}
            payload = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
            response = requests.post(url, data=payload, files=files)
            if response.status_code == 200:
                logger.info("Gif sent to Telegram.")
            else:
                logger.error(f"Failed to send gif, status code: {response.status_code}")
        else:
            logger.error("Failed to read image for Telegram cropping.")
    except Exception as e:
        logger.error(f"Error sending gif to Telegram: {e}")

def send_telegram_gif(caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendAnimation"
    try:
        with open(OUTPUT_GIF, "rb") as gif:
            files = {"animation": (os.path.basename(OUTPUT_GIF), gif, "image/gif")}
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": caption,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=payload, files=files)
            if response.status_code == 200:
                logger.info("GIF sent to Telegram (inline).")
            else:
                logger.error(f"Failed to send GIF, status code: {response.status_code}, response: {response.text}")
    except Exception as e:
        logger.error(f"Error sending GIF to Telegram: {e}")

def download_and_process_image():
    try:
        response = requests.get(IMAGE_URL, timeout=10)
        if response.status_code == 200:
            image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                create_gif_from_activity()
                detect_cats_and_notify(image)
            else:
                logger.error("Failed to decode image for YOLO")
        else:
            logger.error(f"Failed to download image: Status code {response.status_code}")
    except Exception as e:
        logger.error(f"Error downloading image: {e}")

def create_gif_from_activity():

    os.makedirs(TEMP_DIR, exist_ok=True)

    frames = []
    num_frames = int(DURATION / FRAME_INTERVAL)

    for i in range(num_frames):
        try:
            response = requests.get(IMAGE_URL, timeout=5)
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if image is not None:
                    cropped = image[80:, 100:900]
                    frame_path = os.path.join(TEMP_DIR, f"frame_{i}.jpg")
                    cv2.imwrite(frame_path, cropped)
                    frames.append(cropped)
        except Exception as e:
            print(f"Error fetching frame {i}: {e}")

        time.sleep(FRAME_INTERVAL)

    # Convert to RGB and build GIF
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(OUTPUT_GIF, rgb_frames, duration=FRAME_INTERVAL)

    # Clean up temp directory
    shutil.rmtree(TEMP_DIR)

def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        logger.info("Connected to MQTT broker successfully.")
        client.subscribe(TOPIC_MOTION)
        client.subscribe(TOPIC_CAT_ACTIVE)
    else:
        logger.error(f"Failed to connect to MQTT broker, return code {rc}")

def on_message(client, userdata, msg):
    global last_motion_time, active_cats
    try:
        payload = msg.payload.decode("utf-8")
        current_time = time.time()

        if msg.topic == TOPIC_MOTION and payload.strip().upper() == "ON":
            if current_time - last_motion_time >= MOTION_COOLDOWN:
                logger.info("Motion detected. Processing image.")
                time.sleep(3.5)
                download_and_process_image()
            else:
                logger.info("Motion ignored due to cooldown.")

        elif msg.topic == TOPIC_CAT_ACTIVE and payload.strip() == "1":
            if last_motion_time != 0 and (time.time() - last_motion_time) < MOTION_COOLDOWN:
                logger.info("Additional cat activity detected during cooldown.")
                response = requests.get(IMAGE_URL, timeout=10)
                if response.status_code == 200:
                    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
                    if image is not None:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        results = model(image)
                        detected_cats = set()
                        for box in results[0].boxes:
                            confidence = box.conf.item()
                            class_id = int(box.cls.item())
                            label = model.names[class_id]
                            if label in CAT_NAMES:
                                if confidence > 0.7:
                                    detected_cats.add(label)
                                    logger.info(f"Detected {label} with confidence {confidence:.2f}")
                                else:
                                    logger.info(f"Detected {label} with confidence {confidence:.2f}, too low to notify.")
                        new_cats = []
                        for cat in detected_cats:
                            if cat not in active_cats:
                                new_cats.append(cat)

                        if new_cats:
                            logger.info(f"New cat(s) detected during cooldown: {', '.join(new_cats)}")
                            logger.info(f"Active cats now: {', '.join(active_cats)}")
                            active_cats.update(new_cats)
                            filename = os.path.join(SAVE_FOLDER, f"cooldown-detect-{timestamp}.jpg")
                            cv2.imwrite(filename, image)
                            send_telegram_snapshot(f"{', '.join(new_cats)} detected in the litter tray.", filename)
                            last_motion_time = time.time()
                        else:
                            logger.info("Cat(s) already notified in this cooldown. Ignoring.")
    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")

if __name__ == "__main__":
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    logger.info("Cat AI Detection Server Started")

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set(cert_reqs=ssl.CERT_NONE)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, 60)
    logger.info("Listening for motion events via MQTT...")
    client.loop_forever()
