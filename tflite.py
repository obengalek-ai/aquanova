import cv2
import numpy as np
import tensorflow as tf

# Load class names dari dataset kamu
names = ["class1", "class2", "class3"]  # <-- ubah sesuai data.yaml

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="best-fp16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model input shape:", input_details[0]["shape"])  # biasanya [1,640,640,3]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = np.expand_dims(img_rgb, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]["index"], img_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])  # YOLO output
    # ⚠️ Parsing YOLOv5 TFLite output butuh decoding (bounding box, confidence, class)
    # di sini baru raw output
    print("Output raw:", output_data.shape)

    cv2.imshow("TFLite Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
