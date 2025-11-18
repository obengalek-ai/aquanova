# YOLOv5 detection + Arduino (Servo + ESC Integration with Red & Green Ball Filter + Center Line)

import os
import pathlib
import sys
import time
from pathlib import Path

import serial
import torch

# Patch Path untuk Windows
pathlib.PosixPath = pathlib.WindowsPath

# Setup path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# YOLOv5 imports
from ultralytics.utils.plotting import Annotator, colors

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, cv2, non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode

# ----------------- Serial Arduino -----------------
arduino = None
try:
    arduino = serial.Serial("COM3", 9600, timeout=1)  # ganti COM sesuai port
    time.sleep(2)
    print("✅ Arduino connected on COM3")

    # Kirim perintah agar servo ke tengah saat program mulai
    arduino.write(b"CENTER\n")
    print("⚙️ Servo diset ke tengah (startup)")

except Exception as e:
    print("⚠️ Arduino tidak terhubung:", e)


@smart_inference_mode()
def run(
    weights=ROOT / "best.pt",  # model hasil training
    source=0,  # webcam
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    view_img=True,
    classes=None,
    line_thickness=3,
):
    source = str(source)
    webcam = source.isnumeric()

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Warmup
    model.warmup(imgsz=(1, 3, *imgsz))

    # Loop detection
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.float() / 255
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, max_det=max_det)

        for i, det in enumerate(pred):
            if webcam:
                p, im0 = path[i], im0s[i].copy()
            else:
                p, im0 = path, im0s.copy()

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            red_x, red_y = None, None
            green_x, green_y = None, None

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = names[int(cls)]
                    x_center = int((xyxy[0] + xyxy[2]) / 2)
                    y_center = int((xyxy[1] + xyxy[3]) / 2)

                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

                    # Simpan koordinat bola
                    if label.lower() == "bolamerah":
                        red_x, red_y = x_center, y_center
                    elif label.lower() == "bolahijau":
                        green_x, green_y = x_center, y_center

            # ---- Jika kedua bola terdeteksi ----
            if red_x is not None and green_x is not None:
                mid_x = int((red_x + green_x) / 2)
                mid_y = int((red_y + green_y) / 2)

                # Garis antar bola
                cv2.line(im0, (red_x, red_y), (green_x, green_y), (255, 255, 0), 2)

                # Titik tengah di antara bola
                cv2.circle(im0, (mid_x, mid_y), 6, (0, 255, 255), -1)

                # Titik pusat frame kamera
                frame_center_x = im0.shape[1] // 2
                frame_center_y = im0.shape[0] // 2
                cv2.circle(im0, (frame_center_x, frame_center_y), 6, (0, 0, 255), -1)

                # Garis bantu dari pusat kamera ke titik tengah
                cv2.line(im0, (frame_center_x, frame_center_y), (mid_x, mid_y), (0, 255, 0), 2)

                # Kirim koordinat midpoint ke Arduino
                if arduino:
                    data = f"{mid_x},{mid_y}\n"
                    arduino.write(data.encode())
                    print(f"✅ Kirim ke Arduino: Midpoint=({mid_x},{mid_y})")

                    feedback = arduino.readline().decode().strip()
                    if feedback:
                        print("📥 Arduino:", feedback)
            else:
                print("⏸ Tidak kirim (bola merah & hijau belum lengkap)")

            # Tampilkan hasil
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                    print("Keluar...")

                    # Saat keluar, set servo kembali ke tengah
                    if arduino:
                        arduino.write(b"CENTER\n")
                        print("⚙️ Servo diset ke tengah (shutdown)")

                    if hasattr(dataset, "cap") and dataset.cap:
                        dataset.cap.release()
                    cv2.destroyAllWindows()
                    return


def main():
    run()


if __name__ == "__main__":
    main()
