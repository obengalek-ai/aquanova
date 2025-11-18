# optimized_yolov5_arduino.py
# Perubahan: smooth inference + minimal drawing + non-blocking serial + vid_stride support

import os
import sys
import pathlib
from pathlib import Path
import torch
import serial
import time

# Patch Path untuk Windows
pathlib.PosixPath = pathlib.WindowsPath

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
    # timeout kecil agar non-blocking read
    arduino = serial.Serial("COM3", 9600, timeout=0.05)
    time.sleep(2)
    print("✅ Arduino connected on COM3")
except Exception as e:
    print("⚠️ Arduino tidak terhubung:", e)
    arduino = None

@smart_inference_mode()
def run(
    weights=ROOT / "best.pt",        # model hasil training (ganti ke yolov5n.pt untuk lebih cepat)
    source=0,                        # webcam (0)
    data=ROOT / "data/coco128.yaml",
    imgsz=(640, 640),                # ukuran inferensi (sesuaikan; 640 mempertahankan kualitas)
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",                       # "" => otomatis (GPU kalau ada)
    view_img=True,
    classes=None,
    line_thickness=2,
    vid_stride=1,                    # lewati frame input (1 = semua frame, 2 = proses tiap 2nd frame)
    half=False,                      # pakai FP16 jika memungkinkan
    send_threshold_px=8,             # minimal pergeseran mid_x untuk kirim update ke Arduino
):
    source = str(source)
    webcam = source.isnumeric() or source.endswith(".streams")

    # device & model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # dataloader (pakai vid_stride untuk mengurangi beban input)
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # warmup (sesuaikan batch size kecil/1)
    bs = 1
    model.warmup(imgsz=(1, 3, *imgsz))

    last_mid_x = None  # untuk mengurangi kirim serial berulang jika tidak banyak berubah
    last_mid_y = None

    # Loop detection
    for path, im, im0s, vid_cap, s in dataset:
        # convert to tensor & normalize (efisiennya mirip Ultralytics)
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255.0
        if len(im.shape) == 3:
            im = im[None]

        # inference
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, max_det=max_det)

        # proses tiap item di batch (pada webcam biasanya bs = jumlah stream)
        for i, det in enumerate(pred):
            if webcam:
                p, im0 = path[i], im0s[i].copy()
            else:
                p, im0 = path, im0s.copy()

            # Buat annotator sekali per-frame
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            red_x = red_y = None
            green_x = green_y = None

            if len(det):
                # scale boxes ke ukuran im0
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # ITERATE reversed untuk perilaku sama dengan versi awal
                for *xyxy, conf, cls in reversed(det):
                    label = names[int(cls)].lower()
                    x_center = int((xyxy[0] + xyxy[2]) / 2)
                    y_center = int((xyxy[1] + xyxy[3]) / 2)

                    # Hanya gambar box & label untuk bola merah/hijau (kurangi overhead)
                    if label in ("bolamerah", "bola merah", "bola_merah", "redball", "red_ball"):
                        red_x, red_y = x_center, y_center
                        # hanya gambar box & label untuk objek penting
                        annotator.box_label(xyxy, f"{label} {float(conf):.2f}", color=colors(int(cls), True))
                    elif label in ("bolahijau", "bola hijau", "bola_hijau", "greenball", "green_ball"):
                        green_x, green_y = x_center, y_center
                        annotator.box_label(xyxy, f"{label} {float(conf):.2f}", color=colors(int(cls), True))
                    else:
                        # skip drawing untuk kelas lain -> hemat waktu
                        pass

            # Jika kedua bola terdeteksi -> gambar garis/minpoint + kirim ke Arduino
            if (red_x is not None) and (green_x is not None):
                mid_x = int((red_x + green_x) / 2)
                mid_y = int((red_y + green_y) / 2)

                # Gambar garis antar bola dan titik midpoint (minimal drawing)
                cv2.line(im0, (red_x, red_y), (green_x, green_y), (255, 255, 0), 2)
                cv2.circle(im0, (mid_x, mid_y), 6, (0, 255, 255), -1)

                # Frame center & line
                frame_center_x = im0.shape[1] // 2
                frame_center_y = im0.shape[0] // 2
                cv2.circle(im0, (frame_center_x, frame_center_y), 6, (0, 0, 255), -1)
                cv2.line(im0, (frame_center_x, frame_center_y), (mid_x, mid_y), (0, 255, 0), 2)

                # Kirim ke Arduino hanya jika berubah lebih dari threshold pixel
                send = False
                if last_mid_x is None or abs(mid_x - last_mid_x) >= send_threshold_px or abs(mid_y - last_mid_y) >= send_threshold_px:
                    send = True

                if send and arduino:
                    payload = f"{mid_x},{mid_y}\n"
                    try:
                        arduino.write(payload.encode())
                        # baca singkat kalau ada (non-blocking)
                        if arduino.in_waiting:
                            resp = arduino.readline().decode(errors="ignore").strip()
                            if resp:
                                # print feedback sekali — tidak setiap frame
                                print("📥 Arduino:", resp)
                        # update last values
                        last_mid_x, last_mid_y = mid_x, mid_y
                        # print singkat untuk debugging (bisa dikomentari)
                        # print(f"✅ Kirim ke Arduino: Midpoint=({mid_x},{mid_y})")
                    except Exception:
                        # jika gagal kirim, jangan ganggu loop
                        pass
            else:
                # jika tidak lengkap, jangan kirim dan jangan print berulang
                pass

            # Tampilkan hasil (cepat)
            im0 = annotator.result()
            if view_img:
                cv2.imshow("YOLOv5 Detection", im0)
                # gunakan waitKey(1) minimal blocking
                if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                    # release jika ada
                    if webcam and hasattr(dataset, "cap") and dataset.cap:
                        try:
                            dataset.cap.release()
                        except Exception:
                            pass
                    cv2.destroyAllWindows()
                    return

    # end loop
    if view_img:
        cv2.destroyAllWindows()


def main():
    # contoh pemanggilan: bisa modifikasi args manual di sini
    # untuk performa maksimal: gunakan device CUDA, half=True, atau weights kecil (yolov5n.pt)
    run(
        weights=ROOT / "best.pt",
        source=0,
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        device="",         # set "0" jika ingin pakai GPU 0
        view_img=True,
        vid_stride=1,      # ubah ke 2 jika mau proses lebih ringan
        half=False,        # True jika ingin fp16 (jika GPU mendukung)
        send_threshold_px=6
    )

if __name__ == "__main__":
    main()
