from ultralytics import YOLO
import cv2
import paho.mqtt.client as mqtt
import os
from dotenv import load_dotenv

load_dotenv()

#mqtt
def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected with result code {reason_code}")

def on_publish(client, userdata, mid, reason_code, properties):
    print("mid: "+str(mid) + "\n")
    print("client: " + str(client) + "\n")
    print("user data: " + str(userdata) + "\n")

mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.on_connect = on_connect
mqttc.on_publish = on_publish
mqttc.connect(os.getenv("MQTT_SERVER"), int(os.getenv("MQTT_PORT")))

model = YOLO('best.pt')

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    results = model.predict(frame, conf=0.60)
    for result in results:
        for box in result.boxes:
            cls = box.cls.item()
            class_name = model.names[int(cls)]
            if class_name == "helm":
                mqttc.publish(os.getenv("MQTT_TOPIC"), "helm-sign", 0, False)
            elif class_name == "nohelm":
                mqttc.publish(os.getenv("MQTT_TOPIC"), "no-helm-sign", 0, False)
            else:
                mqttc.publish(os.getenv("MQTT_TOPIC"), "helm-sign", 0, False)

    cv2.imshow('frame', results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cam.release()
cv2.destroyAllWindows()