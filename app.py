import os
import glob
import base64
import tempfile
import subprocess

import cv2
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

API_SECRET = os.environ.get("API_SECRET", "")
YTDLP_COOKIES = os.environ.get("YTDLP_COOKIES", "")

app = FastAPI(title="frame-picker")


class AnalyzeReq(BaseModel):
    video_url: str


def _run(cmd, timeout=240):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _cookies_args():
    if YTDLP_COOKIES and os.path.exists(YTDLP_COOKIES):
        return ["--cookies", YTDLP_COOKIES]
    return []


def _duration(url):
    cmd = ["yt-dlp", "--no-warnings", "--skip-download",
           "--print", "%(duration)s"] + _cookies_args() + [url]
    try:
        r = _run(cmd, timeout=60)
        return float(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


@app.get("/")
def health():
    return {"ok": True, "service": "frame-picker"}


@app.post("/analyze")
def analyze(req: AnalyzeReq, x_api_key: str = Header(default="")):
    if API_SECRET and x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="invalid api key")

    url = req.video_url
    with tempfile.TemporaryDirectory() as tmp:
        # 1) Descargar un tramo central del video (donde suele estar el host hablando)
        dur = _duration(url) or 120.0
        start = int(max(0.0, dur * 0.3))
        end = int(min(dur, start + 20))
        section = f"*{start}-{end}"

        out_tmpl = os.path.join(tmp, "seg.%(ext)s")
        dl = ["yt-dlp", "--no-warnings",
              "-f", "bestvideo[height<=1080][ext=mp4]/best[height<=1080]/best",
              "--download-sections", section,
              "--force-keyframes-at-cuts",
              "-o", out_tmpl] + _cookies_args() + [url]
        r = _run(dl)
        segs = glob.glob(os.path.join(tmp, "seg.*"))
        if not segs:
            raise HTTPException(status_code=502,
                                detail=f"yt-dlp download failed: {r.stderr[-400:]}")
        seg = segs[0]

        # 2) Extraer ~12 fotogramas en HD
        framedir = os.path.join(tmp, "frames")
        os.makedirs(framedir, exist_ok=True)
        _run(["ffmpeg", "-y", "-i", seg,
              "-vf", "fps=1/1.5,scale=-2:1080",
              "-frames:v", "12",
              os.path.join(framedir, "f_%03d.jpg")])
        frames = sorted(glob.glob(os.path.join(framedir, "*.jpg")))
        if not frames:
            _run(["ffmpeg", "-y", "-i", seg, "-frames:v", "1",
                  os.path.join(framedir, "f_001.jpg")])
            frames = sorted(glob.glob(os.path.join(framedir, "*.jpg")))
        if not frames:
            raise HTTPException(status_code=500, detail="no frames extracted")

        # 3) Elegir el mejor fotograma con cara (OpenCV)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        best = None
        for fp in frames:
            img = cv2.imread(fp)
            if img is None:
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces) == 0:
                continue
            fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            score = fw * fh
            if best is None or score > best["score"]:
                best = {"score": score, "img": img, "n": int(len(faces)),
                        "box": (int(fx), int(fy), int(fw), int(fh)), "wh": (w, h)}

        if best is None:
            # Sin cara: devolvemos el fotograma central, faces=0
            img = cv2.imread(frames[len(frames) // 2])
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return {"ok": True, "faces": 0, "face": None,
                    "frame_b64": base64.b64encode(buf.tobytes()).decode()}

        fx, fy, fw, fh = best["box"]
        w, h = best["wh"]
        ok, buf = cv2.imencode(".jpg", best["img"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        return {
            "ok": True,
            "faces": best["n"],
            "face": {"cx": (fx + fw / 2) / w, "cy": (fy + fh / 2) / h, "w": fw / w},
            "frame_b64": base64.b64encode(buf.tobytes()).decode(),
        }
