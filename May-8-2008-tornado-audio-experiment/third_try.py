# Full Google Colab cell: download three YouTube videos, extract smooth motion envelopes,
# render preview WAVs at several resonator gains, and save final WAVs to /content/synthesized_audio.
# Paste and run this entire cell in Colab.

# 1) Install dependencies (run once)
!pip install -q yt-dlp opencv-python-headless numpy scipy soundfile tqdm
!apt-get -qq update && apt-get -qq install -y ffmpeg

# 2) Processing code
import os, traceback
import numpy as np
import cv2
import soundfile as sf
from tqdm import tqdm
from yt_dlp import YoutubeDL
from scipy.signal import iirfilter, sosfilt, sosfiltfilt

# ---------- Configuration ----------
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=clFCtNN3GBM",
    "https://www.youtube.com/watch?v=53I3xJdTieM",
    "https://www.youtube.com/watch?v=mADFdEyfnRc",
]

DOWNLOAD_DIR = "/content/downloaded_videos"
OUTPUT_DIR = "/content/synthesized_audio"
PREVIEW_DIR = os.path.join(OUTPUT_DIR, "previews")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Extraction / audio params (tweak to taste)
FRAME_HOP_MS = 200            # coarse sampling reduces choppiness (try 150-300)
ENVELOPE_CUTOFF_HZ = 0.6      # very low cutoff for envelope smoothing (0.3-1.0)
AUDIO_SR = 22050
MAX_CLIP_SECONDS = 120

# Resonator / freight-train params (start values)
RES1_FREQ = 55.0
RES1_Q = 0.9
RES1_GAIN = 4.0

RES2_FREQ = 28.0
RES2_Q = 0.9
RES2_GAIN = 2.5

SUBRUMBLE_FREQ = 14.0
SUBRUMBLE_GAIN = 1.6

# Texture params
GRAIN_LEN_MS = 220
GRAIN_DENSITY = 0.9
TEXTURE_GAIN = 0.5

FINAL_LOWPASS = 8000.0
NOISE_SEED = 1234

# Preview gains to render for quick A/B
PREVIEW_GAINS = [1.0, 2.0, 3.0, 4.0]

# ---------- Helper filters ----------
def lowpass_sos(cutoff, fs, order=4):
    return iirfilter(order, cutoff/(0.5*fs), btype='low', ftype='butter', output='sos')

def bandpass_sos(center, q, fs, order=4):
    bw = center / max(0.001, q)
    low = max(0.1, center - bw/2)
    high = min(0.5*fs - 1.0, center + bw/2)
    return iirfilter(order, [low/(0.5*fs), high/(0.5*fs)], btype='band', ftype='butter', output='sos')

def smooth_env(env, sr_env, cutoff_hz=ENVELOPE_CUTOFF_HZ):
    sos = lowpass_sos(cutoff_hz, sr_env, order=4)
    return sosfiltfilt(sos, env)

# ---------- Download helper ----------
def download_youtube_mp4(url, out_dir):
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
        if not os.path.exists(filename):
            candidate = os.path.join(out_dir, f"{info['id']}.mp4")
            if os.path.exists(candidate):
                filename = candidate
        return filename

# ---------- Envelope extraction (coarse + strong smoothing) ----------
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
    env_smooth = smooth_env(norm, sr_env, cutoff_hz=ENVELOPE_CUTOFF_HZ)
    env_smooth = np.clip(env_smooth, 0.0, 1.0)
    if env_smooth.max() > 0:
        env_smooth = env_smooth / env_smooth.max()
    return times[:len(env_smooth)], env_smooth

# ---------- Improved continuous synthesis ----------
def synthesize_from_envelope(times, env, sr=AUDIO_SR,
                             res1_freq=RES1_FREQ, res1_q=RES1_Q, res1_gain=RES1_GAIN,
                             res2_freq=RES2_FREQ, res2_q=RES2_Q, res2_gain=RES2_GAIN,
                             sub_freq=SUBRUMBLE_FREQ, sub_gain=SUBRUMBLE_GAIN,
                             grain_len_ms=GRAIN_LEN_MS, grain_density=GRAIN_DENSITY,
                             texture_gain=TEXTURE_GAIN, final_lowpass=FINAL_LOWPASS,
                             seed=NOISE_SEED):
    duration = float(times[-1])
    n_audio = int(np.ceil(duration * sr))
    t_audio = np.linspace(0, duration, n_audio, endpoint=False)

    env_audio = np.interp(t_audio, times, env)
    env_audio = smooth_env(env_audio, sr, cutoff_hz=max(0.3, ENVELOPE_CUTOFF_HZ))

    rng = np.random.RandomState(seed)

    # Subrumble: continuous low sine + filtered low noise
    sub_sine = np.sin(2.0 * np.pi * sub_freq * t_audio)
    white = rng.normal(0, 1, size=n_audio)
    sos_lp = lowpass_sos(80.0, sr, order=6)
    low_noise = sosfilt(sos_lp, white)
    subrumble = (0.7 * sub_sine + 0.3 * low_noise) * env_audio * sub_gain

    # Resonant freight-train bands (continuous band-limited noise)
    sos_r1 = bandpass_sos(res1_freq, res1_q, sr, order=4)
    r1 = sosfilt(sos_r1, rng.normal(0, 1, size=n_audio)) * res1_gain * env_audio

    sos_r2 = bandpass_sos(res2_freq, res2_q, sr, order=4)
    r2 = sosfilt(sos_r2, rng.normal(0, 1, size=n_audio)) * res2_gain * env_audio

    # Slight harmonic to add tonal feel
    harmonic = 0.5 * np.sin(2.0 * np.pi * (res1_freq * 1.02) * t_audio) * (env_audio**0.6) * (res1_gain * 0.4)

    # Granular texture: long grains, dense overlap
    grain_len = int(max(1, (grain_len_ms/1000.0) * sr))
    hop = max(1, int(grain_len * (1.0 - grain_density)))
    texture = np.zeros(n_audio, dtype=np.float32)
    win = np.hanning(grain_len)
    grid = np.arange(0, n_audio, hop)
    probs = np.interp(grid, np.arange(n_audio), env_audio)
    for i, pos in enumerate(grid):
        if rng.rand() < probs[i]:
            start = int(pos)
            end = min(n_audio, start + grain_len)
            g = rng.normal(0, 1, size=(end - start,))
            if len(g) > 31:
                g = g - np.convolve(g, np.ones(31)/31, mode='same')
            texture[start:end] += (g * win[:end-start]) * (0.3 + 0.7 * probs[i]) * texture_gain

    combined = subrumble + r1 + r2 + harmonic + texture

    sos_final = lowpass_sos(final_lowpass, sr, order=4)
    combined = sosfiltfilt(sos_final, combined)

    # Soft limiter
    peak = np.max(np.abs(combined)) + 1e-9
    if peak > 0.98:
        combined = combined / peak * 0.98

    # Small feedback tail
    decay = 0.18
    delay = int(0.035 * sr)
    if delay > 0:
        out = np.copy(combined)
        for i in range(delay, len(out)):
            out[i] += decay * out[i - delay]
        combined = out

    maxv = np.max(np.abs(combined)) + 1e-9
    combined = combined / maxv * 0.95
    return combined.astype(np.float32)

# ---------- Main batch processing with previews ----------
errors = []
for url in YOUTUBE_URLS:
    try:
        print("\nDownloading:", url)
        video_path = download_youtube_mp4(url, DOWNLOAD_DIR)
        print("Saved to:", video_path)

        times, env = extract_motion_envelope(video_path, frame_hop_ms=FRAME_HOP_MS, max_seconds=MAX_CLIP_SECONDS)
        print(f"Envelope length: {len(env)} samples, duration ~{times[-1]:.1f}s")

        base = os.path.splitext(os.path.basename(video_path))[0]

        # Render previews for several resonator gains
        print("Rendering previews for res_band_gain values:", PREVIEW_GAINS)
        for g in PREVIEW_GAINS:
            try:
                preview_audio = synthesize_from_envelope(
                    times, env,
                    sr=AUDIO_SR,
                    res1_freq=RES1_FREQ, res1_q=RES1_Q, res1_gain=g,
                    res2_freq=RES2_FREQ, res2_q=RES2_Q, res2_gain=RES2_GAIN,
                    sub_freq=SUBRUMBLE_FREQ, sub_gain=SUBRUMBLE_GAIN,
                    grain_len_ms=GRAIN_LEN_MS, grain_density=GRAIN_DENSITY,
                    texture_gain=TEXTURE_GAIN, final_lowpass=FINAL_LOWPASS,
                    seed=NOISE_SEED
                )
                preview_name = f"{base}_preview_resgain_{g:.1f}.wav"
                preview_path = os.path.join(PREVIEW_DIR, preview_name)
                sf.write(preview_path, preview_audio, AUDIO_SR)
                print("Saved preview:", preview_path)
            except Exception as e:
                print("Preview failed for gain", g, ":", e)

        # Choose final gain (you can change this or pick from previews)
        final_gain = RES1_GAIN  # keep configured RES1_GAIN or set to preferred preview value
        final_audio = synthesize_from_envelope(
            times, env,
            sr=AUDIO_SR,
            res1_freq=RES1_FREQ, res1_q=RES1_Q, res1_gain=final_gain,
            res2_freq=RES2_FREQ, res2_q=RES2_Q, res2_gain=RES2_GAIN,
            sub_freq=SUBRUMBLE_FREQ, sub_gain=SUBRUMBLE_GAIN,
            grain_len_ms=GRAIN_LEN_MS, grain_density=GRAIN_DENSITY,
            texture_gain=TEXTURE_GAIN, final_lowpass=FINAL_LOWPASS,
            seed=NOISE_SEED
        )
        out_wav = os.path.join(OUTPUT_DIR, f"{base}_synth_resgain_{final_gain:.1f}.wav")
        sf.write(out_wav, final_audio, AUDIO_SR)
        print("Saved final synth:", out_wav)

    except Exception as e:
        tb = traceback.format_exc()
        print("Error processing", url, ":", e)
        print(tb)
        errors.append((url, str(e)))

print("\nDone. Files saved in:", OUTPUT_DIR)
print("Preview files in:", PREVIEW_DIR)
!ls -la /content/synthesized_audio || true

if errors:
    print("\nSome videos failed:")
    for u, msg in errors:
        print("-", u, ":", msg)

# Check listed files in each created directory
ls -la downloaded_videos synthesized_audio

# play inline (replace filename with actual file)
from IPython.display import Audio, display
display(Audio('/content/synthesized_audio/mADFdEyfnRc_synth.wav'))

# download to your local machine
#from google.colab import files
#files.download('/content/synthesized_audio/<your_file>_synth.wav')
