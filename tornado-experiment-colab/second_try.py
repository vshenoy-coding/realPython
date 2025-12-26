# Colab: download three YouTube videos, extract motion envelopes, synthesize improved tornado audio,
# and save WAVs to /content/synthesized_audio

# 1) Install dependencies (run once)
!pip install -q yt-dlp opencv-python-headless numpy scipy soundfile tqdm
!apt-get -qq update && apt-get -qq install -y ffmpeg

# 2) Processing cell
import os, sys, traceback
import numpy as np
import cv2
import soundfile as sf
from tqdm import tqdm
from yt_dlp import YoutubeDL
from scipy.signal import iirfilter, sosfilt, sosfiltfilt
from scipy.signal import butter, lfilter

# ---------- Configuration ----------
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=clFCtNN3GBM",
    "https://www.youtube.com/watch?v=53I3xJdTieM",
    "https://www.youtube.com/watch?v=mADFdEyfnRc",
]

DOWNLOAD_DIR = "/content/downloaded_videos"
OUTPUT_DIR = "/content/synthesized_audio"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Audio / extraction params (tweak to taste)
FRAME_HOP_MS = 250        # coarser sampling reduces choppiness
AUDIO_SR = 22050
MAX_CLIP_SECONDS = 120

# Synthesis tuning
SUBRUMBLE_FREQ = 18.0
SUBRUMBLE_GAIN = 1.2
RUMBLE_CUTOFF = 60.0
RES_BAND_FREQ = 40.0
RES_BAND_Q = 0.8
RES_BAND_GAIN = 3.0
GRAIN_LEN_MS = 140
GRAIN_DENSITY = 0.8
TEXTURE_GAIN = 0.6
FINAL_LOWPASS = 8000.0
NOISE_SEED = 42

# ---------- Helpers ----------
def download_youtube_mp4(url, out_dir):
    """Download best mp4 progressive stream using yt-dlp; return local path."""
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        # fallback to id.mp4 if merged filename differs
        if not os.path.exists(filename):
            candidate = os.path.join(out_dir, f"{info['id']}.mp4")
            if os.path.exists(candidate):
                filename = candidate
        return filename

def butter_lowpass_sos(cutoff, fs, order=4):
    sos = iirfilter(order, cutoff/(0.5*fs), btype='low', ftype='butter', output='sos')
    return sos

def bandpass_sos(center, q, fs):
    bw = center / max(0.001, q)
    low = max(0.1, center - bw/2)
    high = min(0.5*fs - 1.0, center + bw/2)
    sos = iirfilter(4, [low/(0.5*fs), high/(0.5*fs)], btype='band', ftype='butter', output='sos')
    return sos

def smooth_envelope(env, sr_env, cutoff=2.0):
    sos = butter_lowpass_sos(cutoff, sr_env, order=4)
    return sosfiltfilt(sos, env)

# ---------- Envelope extraction ----------
def extract_motion_envelope(video_path, frame_hop_ms=FRAME_HOP_MS, max_seconds=MAX_CLIP_SECONDS):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if total_frames > 0 else max_seconds
    duration = min(duration, max_seconds)

    step = frame_hop_ms / 1000.0
    times = np.arange(0.0, duration, step)
    energies = []

    prev_gray = None
    for t in tqdm(times, desc=f"Extracting frames {os.path.basename(video_path)}"):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # downsample for robustness and speed
        small = cv2.resize(gray, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if prev_gray is None:
            energies.append(0.0)
        else:
            diff = cv2.absdiff(small, prev_gray)
            energies.append(float(np.mean(diff)))
        prev_gray = small
    cap.release()

    if len(energies) < 2:
        raise RuntimeError("Not enough frames to extract envelope")

    arr = np.array(energies)
    if arr.max() > arr.min():
        norm = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        norm = arr * 0.0

    sr_env = 1.0 / step
    norm_smooth = smooth_envelope(norm, sr_env, cutoff=1.0)
    norm_smooth = np.clip(norm_smooth, 0.0, 1.0)
    if norm_smooth.max() > 0:
        norm_smooth = norm_smooth / norm_smooth.max()
    return times[:len(norm_smooth)], norm_smooth

# ---------- Improved synthesis ----------
def synthesize_from_envelope(times, env, sr=AUDIO_SR,
                             subrumble_freq=SUBRUMBLE_FREQ,
                             subrumble_gain=SUBRUMBLE_GAIN,
                             rumble_cutoff=RUMBLE_CUTOFF,
                             res_band_freq=RES_BAND_FREQ,
                             res_band_q=RES_BAND_Q,
                             res_band_gain=RES_BAND_GAIN,
                             grain_len_ms=GRAIN_LEN_MS,
                             grain_density=GRAIN_DENSITY,
                             texture_gain=TEXTURE_GAIN,
                             final_lowpass=FINAL_LOWPASS,
                             seed=NOISE_SEED):
    duration = float(times[-1])
    n_audio = int(np.ceil(duration * sr))
    t_audio = np.linspace(0, duration, n_audio, endpoint=False)
    env_audio = np.interp(t_audio, times, env)
    env_audio = smooth_envelope(env_audio, sr, cutoff=2.0)

    rng = np.random.RandomState(seed)

    # Subrumble: low sine + filtered noise
    sub_sine = np.sin(2.0 * np.pi * subrumble_freq * t_audio)
    white = rng.normal(0, 1, size=n_audio)
    sos_lp = butter_lowpass_sos(rumble_cutoff, sr, order=6)
    low_noise = sosfilt(sos_lp, white)
    subrumble = (0.6 * sub_sine + 0.4 * low_noise) * env_audio * subrumble_gain

    # Resonant band
    sos_res = bandpass_sos(res_band_freq, res_band_q, sr)
    res_noise = sosfilt(sos_res, rng.normal(0, 1, size=n_audio))
    wobble = 1.0 + 0.03 * np.sin(2.0 * np.pi * 0.15 * t_audio)
    resonant = res_noise * env_audio * wobble * res_band_gain

    # Granular texture (longer grains, overlap-add)
    grain_len = int(max(1, (grain_len_ms/1000.0) * sr))
    hop = int(grain_len * (1.0 - grain_density))
    if hop < 1:
        hop = 1
    texture = np.zeros(n_audio, dtype=np.float32)
    win = np.hanning(grain_len)
    grid_positions = np.arange(0, n_audio, hop)
    probs = np.interp(grid_positions, np.arange(n_audio), env_audio)
    for i, pos in enumerate(grid_positions):
        if rng.rand() < probs[i]:
            start = int(pos)
            end = min(n_audio, start + grain_len)
            g = rng.normal(0, 1, size=(end - start,))
            if len(g) > 31:
                g = g - np.convolve(g, np.ones(31)/31, mode='same')
            texture[start:end] += (g * win[:end-start]) * (0.4 + 0.6 * probs[i]) * texture_gain

    combined = subrumble + resonant + texture
    sos_final = butter_lowpass_sos(final_lowpass, sr, order=4)
    combined = sosfiltfilt(sos_final, combined)

    # simple feedback tail
    decay = 0.25
    delay_samps = int(0.03 * sr)
    if delay_samps > 0:
        out = np.copy(combined)
        for i in range(delay_samps, len(out)):
            out[i] += decay * out[i - delay_samps]
        combined = out

    maxv = np.max(np.abs(combined)) + 1e-9
    combined = combined / maxv * 0.95
    return combined.astype(np.float32)

# ---------- Main batch processing ----------
errors = []
for url in YOUTUBE_URLS:
    try:
        print("\nDownloading:", url)
        video_path = download_youtube_mp4(url, DOWNLOAD_DIR)
        print("Saved to:", video_path)

        times, env = extract_motion_envelope(video_path, frame_hop_ms=FRAME_HOP_MS, max_seconds=MAX_CLIP_SECONDS)
        print(f"Envelope length: {len(env)} samples, duration ~{times[-1]:.1f}s")

        audio = synthesize_from_envelope(times, env, sr=AUDIO_SR,
                                         subrumble_freq=SUBRUMBLE_FREQ,
                                         subrumble_gain=SUBRUMBLE_GAIN,
                                         rumble_cutoff=RUMBLE_CUTOFF,
                                         res_band_freq=RES_BAND_FREQ,
                                         res_band_q=RES_BAND_Q,
                                         res_band_gain=RES_BAND_GAIN,
                                         grain_len_ms=GRAIN_LEN_MS,
                                         grain_density=GRAIN_DENSITY,
                                         texture_gain=TEXTURE_GAIN,
                                         final_lowpass=FINAL_LOWPASS,
                                         seed=NOISE_SEED)

        base = os.path.splitext(os.path.basename(video_path))[0]
        out_wav = os.path.join(OUTPUT_DIR, f"{base}_synth.wav")
        sf.write(out_wav, audio, AUDIO_SR)
        print("Saved synthesized audio to:", out_wav)
    except Exception as e:
        tb = traceback.format_exc()
        print("Error processing", url, ":", e)
        print(tb)
        errors.append((url, str(e)))

print("\nDone. Files saved in:", OUTPUT_DIR)
print("Contents:")
!ls -la /content/synthesized_audio || true

if errors:
    print("\nSome videos failed:")
    for u, msg in errors:
        print("-", u, ":", msg)

ls -la downloaded_videos synthesized_audio

# play inline (replace filename with actual file)
from IPython.display import Audio, display
display(Audio('/content/synthesized_audio/<your_file>.wav'))

# download to your local machine
#from google.colab import files
#files.download('/content/synthesized_audio/<your_file>_synth.wav')
