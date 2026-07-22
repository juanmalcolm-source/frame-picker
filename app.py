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

# Cookies seguras via variable de entorno (no se guardan en el repo publico).
# Si YTDLP_COOKIES_B64 esta definida, se decodifica a un cookies.txt en disco
# y se usa para autenticar yt-dlp contra YouTube (evita el bloqueo antibot).
_COOKIES_B64 = os.environ.get("YTDLP_COOKIES_B64", "")
if _COOKIES_B64 and not YTDLP_COOKIES:
    try:
        import base64 as _b64
        _cookies_path = "/tmp/cookies.txt"
        with open(_cookies_path, "wb") as _f:
            _f.write(_b64.b64decode(_COOKIES_B64))
        YTDLP_COOKIES = _cookies_path
    except Exception:
        pass

app = FastAPI(title="frame-picker")


class AnalyzeReq(BaseModel):
    video_url: str


def _run(cmd, timeout=240):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _cookies_args():
    if YTDLP_COOKIES and os.path.exists(YTDLP_COOKIES):
        return ["--cookies", YTDLP_COOKIES]
    return []


# Clientes internos de YouTube a probar (algunos esquivan el muro antibot en
# IPs de datacenter). Configurable via YTDLP_CLIENT; por defecto una lista.
YTDLP_CLIENT = os.environ.get("YTDLP_CLIENT", "default,android,ios,tv,web_safari")


def _client_args():
    if YTDLP_CLIENT:
        return ["--extractor-args", f"youtube:player_client={YTDLP_CLIENT}"]
    return []


# Proxy (idealmente residencial) para que YouTube vea una IP no-datacenter.
# Formato: http://usuario:clave@host:puerto  (o socks5://...)
YTDLP_PROXY = os.environ.get("YTDLP_PROXY", "")


def _proxy_args():
    if YTDLP_PROXY:
        return ["--proxy", YTDLP_PROXY]
    return []


def _duration(url):
    cmd = ["yt-dlp", "--no-warnings", "--skip-download",
           "--print", "%(duration)s"] + _cookies_args() + _client_args() + _proxy_args() + [url]
    try:
        r = _run(cmd, timeout=60)
        return float(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


@app.get("/")
def health():
    return {"ok": True, "service": "frame-picker"}


@app.get("/diag")
def diag():
    import numpy
    return {
        "cv2_version": getattr(cv2, "__version__", "?"),
        "cv2_file": getattr(cv2, "__file__", "?"),
        "has_CascadeClassifier": hasattr(cv2, "CascadeClassifier"),
        "has_data": hasattr(cv2, "data"),
        "numpy": numpy.__version__,
        "n_attrs": len(dir(cv2)),
        "cookies_loaded": bool(YTDLP_COOKIES and os.path.exists(YTDLP_COOKIES)),
        "proxy_set": bool(YTDLP_PROXY),
        "client": YTDLP_CLIENT,
    }


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

        def _download(fmt):
            # limpiar restos de intentos previos
            for f in glob.glob(os.path.join(tmp, "seg.*")):
                try:
                    os.remove(f)
                except Exception:
                    pass
            # SIN --force-keyframes-at-cuts: usa el descargador ffmpeg de yt-dlp,
            # que clipa el tramo de forma fiable (el nativo dejaba un mp4 vacio).
            dl = ["yt-dlp", "--no-warnings", "-f", fmt,
                  "--download-sections", section,
                  "-o", out_tmpl] + _cookies_args() + _client_args() + _proxy_args() + [url]
            rr = _run(dl)
            cand = [s for s in glob.glob(os.path.join(tmp, "seg.*"))
                    if not s.endswith(".part") and os.path.getsize(s) > 10000]
            if cand:
                return max(cand, key=lambda s: os.path.getsize(s)), rr
            return None, rr

        # 1er intento: HD (DASH solo-video). 2o: progresivo (un solo fichero).
        seg, r = _download(
            "bestvideo[height<=1080][ext=mp4]/bestvideo[height<=720]"
            "/best[height<=720][ext=mp4]/best")
        if not seg:
            seg, r = _download("best")

        if not seg:
            err = (r.stderr or "").lower()
            if "confirm you" in err or "not a bot" in err or "sign in to confirm" in err:
                reason = "youtube_bot_block"
            elif "cookies" in err and (
                "expired" in err or "invalid" in err
                or "no longer valid" in err or "rotate" in err
            ):
                reason = "cookies_invalid"
            elif "http error 403" in err or "forbidden" in err:
                reason = "forbidden_403"
            elif "video unavailable" in err or "private video" in err:
                reason = "video_unavailable"
            else:
                reason = "download_failed"
            print(f"[analyze] download failed reason={reason}", flush=True)
            raise HTTPException(status_code=502, detail=reason)
        print(f"[analyze] seg={os.path.basename(seg)} "
              f"size={os.path.getsize(seg)}", flush=True)

        # 2) Extraer ~12 fotogramas en HD (solo video, ignoramos audio)
        framedir = os.path.join(tmp, "frames")
        os.makedirs(framedir, exist_ok=True)
        f1 = _run(["ffmpeg", "-y", "-i", seg, "-an", "-map", "0:v:0?",
                   "-vf", "fps=1/1.5,scale=-2:1080",
                   "-frames:v", "12",
                   os.path.join(framedir, "f_%03d.jpg")])
        frames = sorted(glob.glob(os.path.join(framedir, "*.jpg")))
        if not frames:
            # Fallback: sin filtro de fps, sacar unos fotogramas sueltos
            f2 = _run(["ffmpeg", "-y", "-i", seg, "-an", "-map", "0:v:0?",
                       "-vf", "scale=-2:1080", "-frames:v", "6",
                       os.path.join(framedir, "f_%03d.jpg")])
            frames = sorted(glob.glob(os.path.join(framedir, "*.jpg")))
            if not frames:
                print(f"[analyze] no frames. ff1={f1.stderr[-200:]} "
                      f"ff2={f2.stderr[-200:]}", flush=True)
                raise HTTPException(status_code=500, detail="no_frames_extracted")

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
