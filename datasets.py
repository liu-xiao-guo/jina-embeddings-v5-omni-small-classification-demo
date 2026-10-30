"""Generate small synthetic text / image / video / audio sample datasets.

Everything is procedurally generated (no downloads, no licensing issues). Each item has a
natural-language caption so we can test text -> media, media -> text and media -> media search.
"""
import json
import math
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DATA = Path(__file__).parent / "data"
SIZE = 256
SR = 16000


# ---------------------------------------------------------------- images
def _img(bg=(255, 255, 255)):
    im = Image.new("RGB", (SIZE, SIZE), bg)
    return im, ImageDraw.Draw(im)


def _gradient(top, bottom):
    t = np.linspace(0, 1, SIZE)[:, None, None]
    arr = (np.array(top) * (1 - t) + np.array(bottom) * t).astype(np.uint8)
    return Image.fromarray(np.repeat(arr, SIZE, axis=1))


def make_images() -> list[dict]:
    d = DATA / "images"
    d.mkdir(parents=True, exist_ok=True)
    items = []

    def save(name, im, caption):
        p = d / f"{name}.png"
        im.save(p)
        items.append({"modality": "image", "id": f"image:{name}", "title": name.replace("_", " "), "caption": caption, "path": str(p)})

    im, g = _img(); g.ellipse((48, 48, 208, 208), fill=(220, 30, 30))
    save("red_circle", im, "a red circle on a white background")
    im, g = _img(); g.rectangle((56, 56, 200, 200), fill=(30, 70, 220))
    save("blue_square", im, "a blue square on a white background")
    im, g = _img(); g.polygon([(128, 40), (216, 200), (40, 200)], fill=(30, 170, 60))
    save("green_triangle", im, "a green triangle on a white background")
    im, g = _img()
    for i in range(0, SIZE, 32):
        g.rectangle((0, i, SIZE, i + 15), fill=(250, 210, 20))
    save("yellow_stripes", im, "horizontal yellow and white stripes")
    im, g = _img()
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 0:
                g.rectangle((c * 32, r * 32, c * 32 + 31, r * 32 + 31), fill=(0, 0, 0))
    save("checkerboard", im, "a black and white checkerboard pattern")
    save("sunset_gradient", _gradient((255, 140, 0), (90, 30, 140)), "a sunset sky gradient from orange to purple")
    rng = np.random.default_rng(1)
    im, g = _img((8, 10, 40))
    for x, y in rng.integers(0, SIZE, (60, 2)):
        g.ellipse((x, y, x + 3, y + 3), fill=(255, 255, 255))
    g.ellipse((190, 30, 230, 70), fill=(240, 240, 200))
    save("night_sky", im, "a night sky with white stars and a pale moon")
    im, g = _img((135, 200, 250)); g.rectangle((0, 170, SIZE, SIZE), fill=(40, 150, 60)); g.ellipse((180, 20, 230, 70), fill=(255, 220, 0))
    save("sunny_meadow", im, "a sunny day with blue sky, a yellow sun and green grass")
    return items


# ---------------------------------------------------------------- video
def _write_video(path: Path, frame_fn, seconds=3, fps=12, size=192):
    n = seconds * fps
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        im = Image.new("RGB", (size, size), (255, 255, 255))
        frame_fn(ImageDraw.Draw(im), i / n, size)
        pr.stdin.write(np.asarray(im).tobytes())
    pr.stdin.close()
    pr.wait()


def make_videos() -> list[dict]:
    d = DATA / "videos"
    d.mkdir(parents=True, exist_ok=True)
    items = []

    def ball(g, t, s):
        x = int((0.5 + 0.4 * math.sin(2 * math.pi * 2 * t)) * (s - 40)) + 20
        y = int(abs(math.sin(2 * math.pi * 2 * t)) * (s - 60)) + 10
        y = s - 30 - y
        g.ellipse((x - 20, y - 20, x + 20, y + 20), fill=(220, 30, 30))

    def spin(g, t, s):
        a, c, r = 2 * math.pi * t * 2, s / 2, s * 0.3
        g.polygon([(c + r * math.cos(a + k * math.pi / 2 + math.pi / 4) * 1.41, c + r * math.sin(a + k * math.pi / 2 + math.pi / 4) * 1.41) for k in range(4)], fill=(30, 70, 220))

    def grow(g, t, s):
        r = 10 + (s * 0.42) * (0.5 - 0.5 * math.cos(2 * math.pi * t * 2))
        g.ellipse((s / 2 - r, s / 2 - r, s / 2 + r, s / 2 + r), fill=(30, 170, 60))

    def scroll(g, t, s):
        off = int(t * 4 * 32)
        for i in range(-32, s + 32, 32):
            g.rectangle((0, i + off % 32, s, i + off % 32 + 15), fill=(250, 210, 20))

    def flash(g, t, s):
        if int(t * 12) % 2 == 0:
            g.rectangle((0, 0, s, s), fill=(0, 0, 0))

    rng = np.random.default_rng(3)
    flakes = rng.random((40, 2)) * 192
    speeds = 40 + rng.random(40) * 80

    def snow(g, t, s):
        g.rectangle((0, 0, s, s), fill=(30, 40, 80))
        for (x, y0), v in zip(flakes, speeds):
            y = (y0 + v * t * 3) % s
            g.ellipse((x, y, x + 5, y + 5), fill=(255, 255, 255))

    specs = [
        ("bouncing_red_ball", ball, "a red ball bouncing from side to side"),
        ("rotating_blue_square", spin, "a blue square rotating in place"),
        ("pulsing_green_circle", grow, "a green circle growing and shrinking"),
        ("scrolling_stripes", scroll, "yellow stripes scrolling downward"),
        ("flashing_screen", flash, "a screen rapidly flashing black and white"),
        ("falling_snow", snow, "white snowflakes falling on a dark blue background"),
    ]
    for name, fn, caption in specs:
        p = d / f"{name}.mp4"
        _write_video(p, fn)
        items.append({"modality": "video", "id": f"video:{name}", "title": name.replace("_", " "), "caption": caption, "path": str(p)})
    return items


# ---------------------------------------------------------------- audio
def _tone(f, dur, amp=0.5):
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1, np.minimum(t, dur - t) * 50)
    return amp * np.sin(2 * np.pi * f * t) * env


def _write_wav(path: Path, x: np.ndarray):
    x = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def make_audio() -> list[dict]:
    d = DATA / "audio"
    d.mkdir(parents=True, exist_ok=True)
    items = []
    t3 = np.arange(SR * 3) / SR
    rng = np.random.default_rng(5)

    def sweep(f0, f1, dur=3):
        t = np.arange(int(SR * dur)) / SR
        return 0.5 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * dur)))

    siren = np.concatenate([sweep(600, 1400, 1.5), sweep(1400, 600, 1.5)])
    beeps = np.concatenate([np.concatenate([_tone(1000, 0.2), np.zeros(int(SR * 0.2))]) for _ in range(7)])[: SR * 3]
    kick = np.zeros(SR * 3)
    for k in range(6):
        i = int(k * 0.5 * SR); tt = np.arange(int(0.25 * SR)) / SR
        kick[i:i + len(tt)] += 0.9 * np.sin(2 * np.pi * 60 * tt + 8 * np.exp(-tt * 30)) * np.exp(-tt * 18)
    melody = np.concatenate([_tone(f, 0.375) for f in [262, 262, 392, 392, 440, 440, 392, 523]])
    specs = [
        ("pure_tone_440hz", _tone(440, 3), "a steady pure musical tone at 440 hertz"),
        ("c_major_chord", (_tone(262, 3) + _tone(330, 3) + _tone(392, 3)) / 2, "a soft C major chord played on a synthesizer"),
        ("rising_siren", siren, "an emergency siren wailing up and down"),
        ("beeping_alarm", beeps, "a high pitched alarm beeping repeatedly"),
        ("white_noise", 0.25 * rng.standard_normal(SR * 3), "hissing white noise like static or rain"),
        ("low_rumble", 0.6 * np.sin(2 * np.pi * 50 * t3) + 0.2 * rng.standard_normal(SR * 3) * 0.3, "a deep low rumbling bass drone"),
        ("kick_drum", kick, "a rhythmic kick drum beat"),
        ("simple_melody", melody, "a simple children's melody played note by note"),
    ]
    for name, x, caption in specs:
        p = d / f"{name}.wav"
        _write_wav(p, x)
        items.append({"modality": "audio", "id": f"audio:{name}", "title": name.replace("_", " "), "caption": caption, "path": str(p)})
    return items


# ---------------------------------------------------------------- text
EXTRA_TEXT = [
    ("A cat is sleeping on a warm windowsill.", "animals"),
    ("The stock market closed higher after strong earnings reports.", "finance"),
    ("Add two cups of flour and whisk until the batter is smooth.", "cooking"),
    ("The goalkeeper dived to save a late penalty kick.", "sports"),
    ("Heavy rain and thunderstorms are expected across the region tonight.", "weather"),
    ("A fire engine raced through the city with its sirens blaring.", "emergency"),
    ("Snow gently covered the quiet village overnight.", "weather"),
    ("The orchestra tuned their instruments before the concert began.", "music"),
]


def make_text(media_items: list[dict]) -> list[dict]:
    d = DATA / "text"
    d.mkdir(parents=True, exist_ok=True)
    items = []
    for m in media_items:  # one caption document per media item -> text index describes the media
        items.append({"modality": "text", "id": f"text:caption-{m['id'].split(':')[1]}", "title": f"Caption: {m['title']}",
                      "caption": m["caption"].capitalize() + ".", "describes": m["id"]})
    for i, (txt, topic) in enumerate(EXTRA_TEXT):
        items.append({"modality": "text", "id": f"text:extra-{i}", "title": f"Note ({topic})", "caption": txt})
    for it in items:  # also persist each document as a .txt file, like the other modalities
        p = d / f"{it['id'].split(':')[1]}.txt"
        p.write_text(it["caption"] + "\n")
        it["path"] = str(p)
    return items


def generate_all() -> list[dict]:
    media = make_images() + make_videos() + make_audio()
    items = make_text(media) + media
    (DATA / "manifest.json").write_text(json.dumps(items, indent=2))
    return items


def load_manifest() -> list[dict]:
    p = DATA / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else []


if __name__ == "__main__":
    its = generate_all()
    print({m: sum(i["modality"] == m for i in its) for m in ("text", "image", "video", "audio")})
