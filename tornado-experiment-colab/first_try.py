# Colab: install required packages
!pip install -q yt-dlp opencv-python-headless numpy scipy soundfile tqdm

# ffmpeg is usually available in Colab, but ensure it's present
!apt-get -qq update && apt-get -qq install -y ffmpeg

 Colab cell: download the three YouTube videos and synthesize WAVs
import os
import json
import io
import wave
import base64
import numpy as np
import soundfile as sf
import cv2
from scipy.signal import butter, lfilter
from tqdm import tqdm
from yt_dlp import YoutubeDL

# ---------- Configuration ----------
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=clFCtNN3GBM",
    "https://www.youtube.com/watch?v=53I3xJdTieM",
    "https://www.youtube.com/watch?v=mADFdEyfnRc",
]

DOWNLOAD_DIR = "../downloaded_videos"
OUTPUT_DIR = "../synthesized_audio"
AUDIO_SR = 22050
FRAME_HOP_MS = 200            # sample motion every 200 ms
MAX_CLIP_SECONDS = 120      # limit processing length per video (seconds)
RUMBLE_CUTOFF = 40.0
RUMBLE_GAIN = 0.9
TEXTURE_GAIN = 0.6
NOISE_SEED = 42

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Helpers ----------
def download_youtube_mp4(url, out_dir):
    """Download best mp4 progressive stream using yt-dlp; return local path."""
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # merge output to mp4 if needed (yt-dlp will use ffmpeg)
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # yt-dlp returns 'requested_downloads' or 'requested_formats' depending on format selection;
        # the final filename is available in info.get('_filename') or ydl.prepare_filename(info)
        # Use ydl.prepare_filename to be robust:
        filename = ydl.prepare_filename(info)
        # If merged file has different extension, try to find mp4 with same id
        if not os.path.exists(filename):
            # try id.mp4
            candidate = os.path.join(out_dir, f"{info['id']}.mp4")
            if os.path.exists(candidate):
                filename = candidate
        return filename

def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def synthesize_from_envelope(times, env, sr=AUDIO_SR,
                             rumble_cutoff=RUMBLE_CUTOFF,
                             rumble_gain=RUMBLE_GAIN,
                             texture_gain=TEXTURE_GAIN,
                             seed=NOISE_SEED):
    """Return numpy array (float32, -1..1) of synthesized audio from envelope."""
    duration = float(times[-1])
    n_audio = int(np.ceil(duration * sr))
    t_audio = np.linspace(0, duration, n_audio, endpoint=False)
    env_audio = np.interp(t_audio, times, env)
    # smooth envelope
    alpha = 0.01
    for i in range(1, len(env_audio)):
        env_audio[i] = alpha * env_audio[i] + (1 - alpha) * env_audio[i-1]

    rng = np.random.RandomState(seed)
    white = rng.normal(0, 1, size=n_audio)

    # one-pole lowpass for rumble
    fc = float(rumble_cutoff)
    a = np.exp(-2.0 * np.pi * fc / sr)
    b0 = 1.0 - a
    rumble = np.zeros_like(white)
    y = 0.0
    for i, x in enumerate(white):
        y = b0 * x + a * y
        rumble[i] = y
    wobble = 1.0 + 0.02 * np.sin(2.0 * np.pi * 0.2 * t_audio)
    rumble = rumble * env_audio * wobble * rumble_gain

    # texture bursts
    texture = np.zeros_like(rumble)
    burst_len = int(0.05 * sr)
    threshold = np.percentile(env_audio, 60) * 0.6
    indices = np.where(env_audio > threshold)[0]
    step = int(0.02 * sr)
    for i in range(0, len(indices), max(1, step)):
        center = indices[i]
        start = max(0, center - burst_len // 2)
        end = min(n_audio, start + burst_len)
        burst = rng.normal(0, 1, size=(end - start,))
        if len(burst) > 3:
            avg = np.convolve(burst, np.ones(5)/5, mode='same')
            burst = burst - avg
        win = np.hanning(len(burst))
        texture[start:end] += burst * win * (0.5 + 0.5 * env_audio[center]) * texture_gain

    combined = rumble + texture
    # gentle smoothing
    kernel = np.ones(5) / 5.0
    combined = np.convolve(combined, kernel, mode='same')
    maxv = np.max(np.abs(combined)) + 1e-9
    combined = combined / maxv * 0.95
    return combined.astype(np.float32)

# ---------- Video -> envelope ----------
def extract_motion_envelope(video_path, frame_hop_ms=FRAME_HOP_MS, max_seconds=MAX_CLIP_SECONDS):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if total_frames == 0 else total_frames / fps
    duration = min(duration, max_seconds) if duration > 0 else max_seconds

    # sample times
    step = frame_hop_ms / 1000.0
    times = np.arange(0.0, duration, step)
    energies = []

    # read first frame
    prev_gray = None
    for t in tqdm(times, desc=f"Extracting frames from {os.path.basename(video_path)}"):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            # if seek failed, break
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            energies.append(0.0)
        else:
            # mean absolute difference
            diff = cv2.absdiff(gray, prev_gray)
            energies.append(float(np.mean(diff)))
        prev_gray = gray
    cap.release()

    if len(energies) < 2:
        raise RuntimeError("Not enough frames to extract envelope")

    arr = np.array(energies)
    # normalize 0..1
    if arr.max() > arr.min():
        norm = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        norm = arr * 0.0
    return times[:len(norm)], norm

# ---------- Main batch processing ----------
for url in YOUTUBE_URLS:
    try:
        print("\nDownloading:", url)
        video_path = download_youtube_mp4(url, DOWNLOAD_DIR)
        print("Saved to:", video_path)

        # extract envelope
        times, env = extract_motion_envelope(video_path, frame_hop_ms=FRAME_HOP_MS, max_seconds=MAX_CLIP_SECONDS)
        print(f"Envelope length: {len(env)} samples, duration ~{times[-1]:.1f}s")

        # synthesize audio
        audio = synthesize_from_envelope(times, env, sr=AUDIO_SR,
                                         rumble_cutoff=RUMBLE_CUTOFF,
                                         rumble_gain=RUMBLE_GAIN,
                                         texture_gain=TEXTURE_GAIN,
                                         seed=NOISE_SEED)

        # save WAV
        base = os.path.splitext(os.path.basename(video_path))[0]
        out_wav = os.path.join(OUTPUT_DIR, f"{base}_synth.wav")
        sf.write(out_wav, audio, AUDIO_SR)
        print("Saved synthesized audio to:", out_wav)
    except Exception as e:
        print("Error processing", url, ":", e)

print("\nAll done. Synthesized files are in the folder:", OUTPUT_DIR)

# play inline (replace filename with actual file)
from IPython.display import Audio, display
display(Audio('/content/synthesized_audio/<your_file>.wav'))

# download to your local machine
#from google.colab import files
#files.download('/content/synthesized_audio/<your_file>_synth.wav')
