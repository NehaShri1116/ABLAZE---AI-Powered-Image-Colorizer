import os
from pathlib import Path
import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None
    print("torch not available; enhancement models requiring torch will be skipped.")

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"

PROT = MODELS_DIR / "colorization_deploy_v2.prototxt"
MODEL = MODELS_DIR / "colorization_release_v2.caffemodel"
PTS = MODELS_DIR / "pts_in_hull.npy"

print("Using colorization models from:", MODELS_DIR)
print("Resolved model files:", PROT, MODEL, PTS)

net = None
if PROT.exists() and MODEL.exists() and PTS.exists():
    try:
        print("Loading Zhang colorizer (fallback)...")
        net = cv2.dnn.readNetFromCaffe(str(PROT), str(MODEL))
        pts_np = np.load(str(PTS))
        pts_np = pts_np.transpose().reshape(2, 313, 1, 1)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [pts_np.astype(np.float32)]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
            np.full([1, 313], 2.606, np.float32)
        ]
    except Exception as e:
        print("Failed to load Zhang model:", e)
        net = None
else:
    print("Zhang model files missing; place prototxt, caffemodel, pts_in_hull.npy in:", MODELS_DIR)

DEOLDIFY_AVAILABLE = False
GFPGAN_AVAILABLE = False
REALESRGAN_AVAILABLE = False

try:
    from deoldify import device
    from deoldify.device_id import DeviceId
    from deoldify.visualize import get_image_colorizer
    DEOLDIFY_AVAILABLE = True
    print("DeOldify import OK")
except Exception:
    DEOLDIFY_AVAILABLE = False

try:
    from gfpgan import GFPGANer
    GFPGAN_AVAILABLE = True
    print("GFPGAN import OK")
except Exception:
    GFPGAN_AVAILABLE = False

try:
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
    print("RealESRGAN import OK")
except Exception:
    REALESRGAN_AVAILABLE = False

DEOLDIFY_ART_WEIGHT = MODELS_DIR / "ColorizeArtistic_gen.pth"
DEOLDIFY_STABLE_WEIGHT = MODELS_DIR / "ColorizeStable_gen.pth"
GFPGAN_WEIGHT = MODELS_DIR / "GFPGANv1.3.pth"
REALESRGAN_WEIGHT = MODELS_DIR / "RealESRGAN_x4plus.pth"

# ---------- helpers ----------
def remove_yellow_cast(img, strength=6):
    """
    Gentle yellow cast removal. Lower default strength to avoid blue shift.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    # subtract small value from B (reduces yellow bias gently)
    B = cv2.subtract(B, int(strength))
    lab = cv2.merge([L, A, B])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def blend_with_mask(orig, restored, alpha=0.55):
    if orig.shape != restored.shape:
        restored = cv2.resize(restored, (orig.shape[1], orig.shape[0]))
    return cv2.addWeighted(restored, alpha, orig, 1 - alpha, 0)

def detect_faces(img):
    try:
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        fc = cv2.CascadeClassifier(cascade)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = fc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30,30))
        return faces
    except Exception:
        return ()

# ---------- deoldify setup ----------
_deoldify_colorizer = None

def ensure_deoldify(artistic=True, device_pref="cuda"):
    global _deoldify_colorizer
    if not DEOLDIFY_AVAILABLE:
        raise RuntimeError("DeOldify unavailable")

    try:
        if device_pref == "cuda" and (torch is not None and torch.cuda.is_available()):
            device.set(device=DeviceId.GPU0)
        else:
            device.set(device=DeviceId.CPU)
    except Exception:
        pass

    desired_w = DEOLDIFY_ART_WEIGHT if artistic else DEOLDIFY_STABLE_WEIGHT
    desired_w = Path(desired_w)
    if not desired_w.exists():
        raise RuntimeError(f"DeOldify weight not found: {desired_w}")

    models_dir = desired_w.parent
    candidate1 = models_dir
    candidate2 = models_dir.parent

    def would_find_weight(root_folder):
        path_try = Path(root_folder) / "models" / desired_w.name
        return path_try.exists()

    if would_find_weight(candidate1):
        root_folder = candidate1
    elif would_find_weight(candidate2):
        root_folder = candidate2
    else:
        root_folder = candidate2

    if _deoldify_colorizer is None:
        _deoldify_colorizer = get_image_colorizer(root_folder=root_folder, artistic=artistic)

    return _deoldify_colorizer

# ---------- small exposure helper ----------
def mild_exposure_boost(img, clip_limit=1.4, face_mask=None):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    v_boost = clahe.apply(v)

    if face_mask is not None:
        mask = face_mask.astype(np.float32) / 255.0
        mask = cv2.GaussianBlur(mask, (31,31), 0)
        inside = (mask * v_boost + (1.0 - mask) * v).astype(np.uint8)
        v_final = inside
    else:
        v_final = v_boost

    hsv = cv2.merge([h, s, v_final])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_face_restore(img, face_alpha=0.35, preserve_face_brightness=True):
    out = img.copy()
    gfpgan_applied = False
    if GFPGAN_AVAILABLE and GFPGAN_WEIGHT.exists():
        try:
            restorer = GFPGANer(model_path=str(GFPGAN_WEIGHT), upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
            _, _, restored = restorer.enhance(out, has_aligned=False)
            gfpgan_applied = True
        except Exception as e:
            print("GFPGAN failed:", e)
            gfpgan_applied = False

    if not gfpgan_applied:
        return out

    faces = detect_faces(out)
    if len(faces) == 0:
        return blend_with_mask(out, restored, alpha=face_alpha)

    mask = np.zeros((out.shape[0], out.shape[1]), dtype=np.uint8)
    for (x,y,w,h) in faces:
        pad_w = int(w * 0.22)
        pad_h = int(h * 0.28)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(out.shape[1], x + w + pad_w)
        y1 = min(out.shape[0], y + h + pad_h)
        cv2.ellipse(mask, ((x0+x1)//2, (y0+y1)//2), ((x1-x0)//2, (y1-y0)//2), 0, 0, 360, 255, -1)

    mask = cv2.GaussianBlur(mask, (31,31), 0)

    face_alpha_face = float(face_alpha)
    face_alpha_global = max(face_alpha, 0.5)

    alpha_map = (mask.astype(np.float32)/255.0) * face_alpha_face + (1.0 - mask.astype(np.float32)/255.0) * face_alpha_global
    alpha_map = np.dstack([alpha_map]*3)

    out_f = out.astype(np.float32)
    restored_f = restored.astype(np.float32)

    blended = (restored_f * alpha_map + out_f * (1.0 - alpha_map)).astype(np.uint8)

    if preserve_face_brightness:
        blended = mild_exposure_boost(blended, clip_limit=1.3, face_mask=mask)

    return blended

# ---------- Real-ESRGAN (unchanged from your file) ----------
def apply_sr(img):
    if not REALESRGAN_AVAILABLE or not REALESRGAN_WEIGHT.exists() or torch is None:
        return img
    try:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = RealESRGANer(device=device_name, scale=4)
        except Exception:
            print("RealESRGAN: constructor mismatch; skipping SR.")
            return img

        try:
            if hasattr(model, "load_weights"):
                model.load_weights(str(REALESRGAN_WEIGHT))
            elif hasattr(model, "load") and callable(getattr(model, "load")):
                model.load(str(REALESRGAN_WEIGHT))
        except Exception as e:
            print("RealESRGAN weight load failed; skipping SR.", e)
            return img

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sr = model.enhance(rgb)
        out = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
        print("RealESRGAN applied")
        return out
    except Exception as e:
        print("RealESRGAN failed:", e)
        return img

# ---------- Zhang colorizer (unchanged) ----------
def zhang_colorize(img):
    if net is None:
        raise RuntimeError("Zhang model not loaded")
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    L_rs = cv2.resize(L, (224, 224))
    L_rs -= 50
    net.setInput(cv2.dnn.blobFromImage(L_rs))
    ab = net.forward()[0].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    lab_out = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)
    colorized = np.clip(colorized * 255, 0, 255).astype("uint8")
    return colorized

# ---------- main wrapper (small changes: use gentler defaults) ----------
def colorize_image(img_path, mode='auto', render_factor=35, do_enhance=True):
    img_path_p = Path(img_path)
    img = cv2.imread(str(img_path_p))
    if img is None:
        raise ValueError("Image cannot be loaded: " + str(img_path_p))

    orig_h, orig_w = img.shape[:2]

    MAX_WORK = 900
    if max(orig_h, orig_w) > MAX_WORK:
        scale = MAX_WORK / float(max(orig_h, orig_w))
        ws_w = int(orig_w * scale)
        ws_h = int(orig_h * scale)
        img_work = cv2.resize(img, (ws_w, ws_h), interpolation=cv2.INTER_AREA)
    else:
        img_work = img.copy()

    colorized_work = None
    use_deoldify = (mode == 'deoldify') or (mode == 'auto' and DEOLDIFY_AVAILABLE)
    if use_deoldify:
        try:
            print("Using DeOldify for colorization...")
            artistic = True
            colorizer = ensure_deoldify(artistic=artistic)

            deoldify_out = colorizer.get_transformed_image(path=str(img_path_p),
                                                          render_factor=render_factor,
                                                          watermarked=False,
                                                          post_process=True)
            print("DeOldify returned type:", type(deoldify_out))

            if isinstance(deoldify_out, (str, Path)):
                out_path = str(deoldify_out)
                colorized_full = cv2.imread(out_path)
                if colorized_full is None:
                    raise RuntimeError(f"DeOldify returned path but cv2.imread failed: {out_path}")
            elif 'PIL' in type(deoldify_out).__module__:
                pil_img = deoldify_out
                arr = np.array(pil_img)
                if arr.ndim == 3 and arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                colorized_full = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif isinstance(deoldify_out, np.ndarray):
                arr = deoldify_out
                if arr.dtype == np.float32 or arr.dtype == np.float64:
                    arr = np.clip(arr * 255.0, 0, 255).astype('uint8')
                if arr.ndim == 3 and arr.shape[2] == 3:
                    colorized_full = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                else:
                    colorized_full = arr
            else:
                raise RuntimeError(f"Unrecognized DeOldify return type: {type(deoldify_out)}")

            colorized_work = cv2.resize(colorized_full, (img_work.shape[1], img_work.shape[0]), interpolation=cv2.INTER_AREA)

        except Exception as e:
            print("DeOldify failed:", e)
            colorized_work = None

    if colorized_work is None:
        if net is None:
            raise RuntimeError("No colorization model available (DeOldify failed and Zhang unavailable).")
        print("Using Zhang fallback for colorization...")
        colorized_work = zhang_colorize(img_work)

    faces = detect_faces(colorized_work)
    face_mask = None
    if len(faces) > 0:
        mask = np.zeros((colorized_work.shape[0], colorized_work.shape[1]), dtype=np.uint8)
        for (x,y,w,h) in faces:
            pad_w = int(w * 0.22)
            pad_h = int(h * 0.28)
            x0 = max(0, x - pad_w)
            y0 = max(0, y - pad_h)
            x1 = min(colorized_work.shape[1], x + w + pad_w)
            y1 = min(colorized_work.shape[0], y + h + pad_h)
            cv2.ellipse(mask, ((x0+x1)//2, (y0+y1)//2), ((x1-x0)//2, (y1-y0)//2), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (21,21), 0)
        face_mask = mask

    # gentler yellow-cast correction
    colorized_work = remove_yellow_cast(colorized_work, strength=6)
    colorized_work = mild_exposure_boost(colorized_work, clip_limit=1.4, face_mask=face_mask)

    if do_enhance:
        try:
            restored = apply_face_restore(colorized_work, face_alpha=0.35, preserve_face_brightness=True)
            colorized_work = blend_with_mask(colorized_work, restored, alpha=0.6)
        except Exception as e:
            print("Face restoration failed:", e)

    final = cv2.resize(colorized_work, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    if do_enhance:
        try:
            final = apply_sr(final)
        except Exception as e:
            print("SR failed:", e)

    final = mild_exposure_boost(final, clip_limit=1.3, face_mask=None)
    return final

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python colorize.py input.jpg [output.jpg]")
        sys.exit(1)
    inp = sys.argv[1]
    out = colorize_image(inp, mode='auto', render_factor=35, do_enhance=False)
    outpath = sys.argv[2] if len(sys.argv) > 2 else "out_colorized.png"
    cv2.imwrite(outpath, out)
    print("Saved", outpath)
