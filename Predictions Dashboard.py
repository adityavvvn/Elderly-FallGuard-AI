import os
import io
import base64
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for server rendering
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, flash
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import threading
from dotenv import load_dotenv
load_dotenv()

try:
    from twilio.rest import Client as TwilioClient
except Exception:
    TwilioClient = None


# --------------------------------------
# Flask setup
# --------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")


# --------------------------------------
# Configuration
# --------------------------------------
# Default model/scaler locations (override with env vars if needed)
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "bilstm_attention_prefall.h5"))
SCALER_PATH = os.environ.get("SCALER_PATH", os.path.join("models", "scaler_prefall.pkl"))

# Windowing configuration (must match training)
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 200))
STEP_SIZE = int(os.environ.get("STEP_SIZE", 100))
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", 200))
LOWPASS_CUTOFF = float(os.environ.get("LOWPASS_CUTOFF", 30))

# Notifications (optional)
NOTIFY_ENABLED = os.environ.get("NOTIFY_ENABLED", "false").lower() in ("1", "true", "yes", "on")
NOTIFY_LEVEL = os.environ.get("NOTIFY_LEVEL", "danger").lower()  # danger|warn|both
NOTIFY_VIA = os.environ.get("NOTIFY_VIA", "whatsapp").lower()   # whatsapp|sms
EMERGENCY_TO = [p.strip() for p in os.environ.get("EMERGENCY_TO", "").split(",") if p.strip()]
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM")  # Verified Twilio number
TWILIO_VOICE_ENABLED = os.environ.get("TWILIO_VOICE_ENABLED", "false").lower() in ("1","true","yes","on")
TWILIO_CONTENT_SID = os.environ.get("TWILIO_CONTENT_SID")  # Optional: Twilio Content Template SID
TWILIO_CONTENT_VARS = os.environ.get("TWILIO_CONTENT_VARS")  # Optional JSON string for variables


# --------------------------------------
# Loss used during training (custom object)
# --------------------------------------
def focal_loss(gamma=2.0, alpha=[0.5, 1.0, 1.5]):
    alpha_const = tf.constant(alpha, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true_flat = tf.cast(tf.reshape(y_true, (-1,)), tf.int32)
        y_true_one_hot = tf.one_hot(y_true_flat, depth=tf.shape(y_pred)[-1])
        y_pred_clipped = tf.clip_by_value(y_pred, 1e-9, 1.0)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred_clipped, axis=-1)
        alpha_factor = tf.reduce_sum(y_true_one_hot * alpha_const, axis=-1)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        loss = -alpha_factor * modulating_factor * tf.math.log(p_t)
        return tf.reduce_mean(loss)

    return loss_fn


# --------------------------------------
# Safe model/scaler loading
# --------------------------------------
_model = None
_scaler = None


def load_model_and_scaler_once():
    global _model, _scaler
    if _model is not None and _scaler is not None:
        return _model, _scaler

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at '{SCALER_PATH}'.")

    try:
        _model = load_model(MODEL_PATH, custom_objects={"focal_loss": focal_loss})
    except Exception:
        # Load without compilation, then compile manually
        _model = load_model(MODEL_PATH, compile=False)
        _model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss=focal_loss(gamma=2.0, alpha=[0.5, 1.0, 1.5]),
            metrics=["accuracy"],
        )

    with open(SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)

    return _model, _scaler


# --------------------------------------
# Preprocessing utilities (aligned with training)
# --------------------------------------
def parse_txt_file_from_bytes(file_bytes: bytes) -> np.ndarray:
    rows = []
    for raw_line in file_bytes.splitlines():
        try:
            line = raw_line.decode(errors="ignore")
        except Exception:
            continue
        line = line.strip().rstrip(";")
        if not line:
            continue
        tokens = [t.strip() for t in line.split(",") if t.strip() != ""]
        if len(tokens) != 9:
            continue
        try:
            vals = [float(x) for x in tokens]
        except Exception:
            continue
        rows.append(vals)
    if len(rows) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return np.array(rows, dtype=np.float32)


def butter_lowpass_filter(data: np.ndarray, fs: int = SAMPLE_RATE, cutoff: float = LOWPASS_CUTOFF, order: int = 4) -> np.ndarray:
    if data.shape[0] < (order * 3):
        return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    try:
        return filtfilt(b, a, data, axis=0)
    except Exception:
        return data


def create_windows(trial_array: np.ndarray, window_size: int = WINDOW_SIZE, step_size: int = STEP_SIZE, apply_filter: bool = True) -> np.ndarray:
    if trial_array.size == 0:
        return np.empty((0, window_size, 0))
    if apply_filter:
        trial_array = butter_lowpass_filter(trial_array, fs=SAMPLE_RATE, cutoff=LOWPASS_CUTOFF)
    n = trial_array.shape[0]
    windows = []
    for start in range(0, max(0, n - window_size + 1), step_size):
        end = start + window_size
        if end <= n:
            windows.append(trial_array[start:end])
    if len(windows) == 0:
        return np.empty((0, window_size, trial_array.shape[1]))
    return np.stack(windows, axis=0)


def scale_windows(windows: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    if windows.size == 0:
        return windows
    num_windows, win_len, num_channels = windows.shape
    flattened = windows.reshape(num_windows, -1)
    scaled = scaler.transform(flattened)
    return scaled.reshape(num_windows, win_len, num_channels)


# --------------------------------------
# Plot helpers (return base64-encoded PNG)
# --------------------------------------
def plot_probabilities(pred_probs: np.ndarray) -> str:
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 4))
    plt.plot(pred_probs[:, 0], label="Normal", lw=2)
    plt.plot(pred_probs[:, 1], label="Pre-Fall", lw=2)
    plt.plot(pred_probs[:, 2], label="Fall", lw=2)
    plt.xlabel("Window Index (time progression)")
    plt.ylabel("Probability")
    plt.title("Predicted Probabilities Across Windows")
    plt.legend()
    plt.grid(True)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_timeline(pred_classes: np.ndarray) -> str:
    plt.figure(figsize=(12, 2))
    plt.imshow([pred_classes], aspect="auto", cmap="viridis", extent=[0, len(pred_classes), 0, 1])
    plt.yticks([])
    plt.xlabel("Window Index")
    plt.title("Predicted Class Timeline (0=Normal,1=Pre-Fall,2=Fall)")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# --------------------------------------
# Notifications
# --------------------------------------
def _twilio_client_or_none():
    if not (NOTIFY_ENABLED and TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM and EMERGENCY_TO):
        return None
    if TwilioClient is None:
        return None
    try:
        return TwilioClient(TWILIO_SID, TWILIO_TOKEN)
    except Exception:
        return None


def _should_notify(level: str) -> bool:
    if not NOTIFY_ENABLED:
        return False
    if NOTIFY_LEVEL == "both":
        return level in ("warn", "danger")
    if NOTIFY_LEVEL == "warn":
        return level == "warn"
    return level == "danger"


def notify_contacts_async(level: str, summary: dict):
    client = _twilio_client_or_none()
    if client is None or not _should_notify(level):
        return
    def _task():
        try:
            label_counts = ", ".join(f"{k}:{v}" for k, v in summary.items())
            message_text = f"Fall Detection Alert ({level.upper()}): {label_counts}"
            for dest in EMERGENCY_TO:
                try:
                    from_num = TWILIO_FROM
                    to_num = dest
                    if NOTIFY_VIA == "whatsapp":
                        from_num = f"whatsapp:{from_num}"
                        to_num = f"whatsapp:{to_num}"
                    if TWILIO_CONTENT_SID:
                        kwargs = {"from_": from_num, "to": to_num, "content_sid": TWILIO_CONTENT_SID}
                        if TWILIO_CONTENT_VARS:
                            kwargs["content_variables"] = TWILIO_CONTENT_VARS
                        client.messages.create(**kwargs)
                    else:
                        client.messages.create(body=message_text, from_=from_num, to=to_num)
                except Exception:
                    pass
                if TWILIO_VOICE_ENABLED:
                    try:
                        twiml = f"<Response><Say voice='Polly.Matthew'>Emergency. {message_text}. Please check immediately.</Say></Response>"
                        client.calls.create(from_=TWILIO_FROM, to=dest, twiml=twiml)
                    except Exception:
                        pass
        except Exception:
            pass
    threading.Thread(target=_task, daemon=True).start()


# --------------------------------------
# Routes
# --------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part in request.")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.")
            return redirect(request.url)

        try:
            model, scaler = load_model_and_scaler_once()
        except Exception as e:
            flash(f"Error loading model/scaler: {e}")
            return redirect(request.url)

        file_bytes = file.read()
        trial_data = parse_txt_file_from_bytes(file_bytes)
        if trial_data.shape[0] == 0:
            flash("Empty or unreadable file. Ensure SisFall .txt format with 9 columns per line.")
            return redirect(request.url)

        windows = create_windows(trial_data, window_size=WINDOW_SIZE, step_size=STEP_SIZE, apply_filter=True)
        if windows.shape[0] == 0:
            flash("File too short for the configured window size.")
            return redirect(request.url)

        try:
            windows_scaled = scale_windows(windows, scaler)
        except Exception as e:
            flash(f"Scaling failed: {e}")
            return redirect(request.url)

        try:
            pred_probs = model.predict(windows_scaled, verbose=0)
        except Exception as e:
            flash(f"Model inference failed: {e}")
            return redirect(request.url)

        pred_classes = np.argmax(pred_probs, axis=1)
        label_map = {0: "Normal", 1: "Pre-Fall", 2: "Fall"}

        unique, counts = np.unique(pred_classes, return_counts=True)
        summary = {label_map[k]: int(v) for k, v in zip(unique, counts)}

        alert = None
        if 2 in pred_classes:
            alert = "Detected potential FALL sequence in this file!"
            notify_contacts_async("danger", summary)
        elif 1 in pred_classes:
            alert = "Pre-Fall motion detected — caution advised."
            notify_contacts_async("warn", summary)
        else:
            alert = "No fall or pre-fall detected. Activity appears normal."

        try:
            plot_probs_b64 = plot_probabilities(pred_probs)
            plot_timeline_b64 = plot_timeline(pred_classes)
        except Exception as e:
            flash(f"Plotting failed: {e}")
            return redirect(request.url)

        return render_template(
            "result.html",
            windows_count=windows.shape[0],
            summary=summary,
            alert=alert,
            plot_probs_b64=plot_probs_b64,
            plot_timeline_b64=plot_timeline_b64,
        )

    return render_template("index.html")


if __name__ == "__main__":
    # Ensure models directory exists hint for users
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)


