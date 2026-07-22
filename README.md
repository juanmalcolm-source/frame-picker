# frame-picker

Servicio mínimo (FastAPI) que, dado un enlace de YouTube, descarga un tramo del
vídeo, extrae fotogramas en HD y devuelve el mejor fotograma con cara (el host),
junto con la caja de la cara para reencuadrar.

## Contrato

`POST /analyze`

- Header: `x-api-key: <API_SECRET>`
- Body: `{ "video_url": "https://www.youtube.com/watch?v=XXXX" }`
- Respuesta:
  ```json
  {
    "ok": true,
    "faces": 1,
    "face": { "cx": 0.42, "cy": 0.38, "w": 0.18 },
    "frame_b64": "<jpeg base64>"
  }
  ```

`GET /` → healthcheck.

## Variables de entorno

- `API_SECRET` (obligatoria): protege el endpoint.
- `YTDLP_COOKIES` (opcional): ruta a un `cookies.txt` del navegador. Si YouTube
  bloquea IPs de datacenter ("confirm you're not a bot"), sube el archivo y
  define `YTDLP_COOKIES=/app/cookies.txt`.

## Cómo funciona

1. `yt-dlp` baja ~20 s del tramo central del vídeo (donde suele estar el host).
2. `ffmpeg` saca ~12 fotogramas en 1080p.
3. `OpenCV` detecta caras y elige el fotograma con la cara más grande/clara.
4. Devuelve ese fotograma (base64) + la caja de la cara normalizada (0–1).

## Local

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

## Docker

```bash
docker build -t frame-picker .
docker run -p 8080:8080 -e API_SECRET=cambia-esto frame-picker
```
