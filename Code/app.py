from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import time
import numpy as np
from colorize import colorize_image

app = Flask(__name__)

UPLOAD = "static/uploads/"
RESULTS = "static/results/"

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)


def apply_adjustments(bgr_img, contrast=1.0, sharpness=1.0, denoise=0.0):
    """
    Apply denoise -> contrast -> sharpness to a BGR uint8 image and return BGR uint8 result.

    - contrast: multiplicative (1.0 = unchanged)
    - sharpness: 1.0 = unchanged; >1 sharpen; <1 soften
    - denoise: 0.0..1.0 where 0 = off, 1 = strong. Server uses OpenCV fastNlMeansDenoisingColored
    """
    img = bgr_img.copy()

    # 1) DENOISE (server-side): map denoise (0..1) to h param (0..30)
    try:
        d = float(denoise)
    except Exception:
        d = 0.0
    d = max(0.0, min(1.0, d))
    if d > 1e-4:
        # h controls filter strength; tune as needed
        h = d * 30.0        # color component strength
        hColor = d * 30.0   # same for color
        try:
            # cv2.fastNlMeansDenoisingColored expects BGR uint8
            img = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, 7, 21)
        except Exception:
            # if the function fails (older OpenCV), fallback to Gaussian blur mild
            k = max(1, int(d * 3) * 2 + 1)
            img = cv2.GaussianBlur(img, (k, k), 0)

    # 2) Contrast
    try:
        alpha = float(contrast)
    except Exception:
        alpha = 1.0
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

    # 3) Sharpness via unsharp-like approach
    try:
        s = float(sharpness)
    except Exception:
        s = 1.0
    amount = s - 1.0
    if abs(amount) < 1e-4:
        return img

    # sigma adapted by image size
    himg, wimg = img.shape[:2]
    sigma = max(0.8, min(3.0, max(himg, wimg) / 1000.0 * 1.2))
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

    if amount > 0:
        sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
        sharpened = np.clip(sharpened, 0, 255).astype('uint8')
        return sharpened
    else:
        factor = max(0.0, 1.0 + amount)
        softened = cv2.addWeighted(img, factor, blurred, 1.0 - factor, 0)
        softened = np.clip(softened, 0, 255).astype('uint8')
        return softened


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/colorizer")
def colorizer():
    return render_template("app.html", input_image=None, output_image=None, ts=int(time.time()))


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return redirect(url_for('colorizer'))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for('colorizer'))

    filename = file.filename
    file_path = os.path.join(UPLOAD, filename)
    file.save(file_path)

    # read form values (safe fallbacks)
    try:
        contrast = float(request.form.get("contrast", 1.0))
    except Exception:
        contrast = 1.0
    try:
        sharpness = float(request.form.get("sharpness", 1.0))
    except Exception:
        sharpness = 1.0
    try:
        denoise = float(request.form.get("denoise", 0.0))
    except Exception:
        denoise = 0.0

    # colorize_image called with defaults (mode auto, do_enhance default True)
    try:
        result = colorize_image(file_path)
    except Exception as e:
        return f"Processing failed: {e}"

    # apply server-side denoise/contrast/sharpness
    try:
        result_adj = apply_adjustments(result, contrast=contrast, sharpness=sharpness, denoise=denoise)
    except Exception as e:
        print("Adjustment failed:", e)
        result_adj = result

    output_filename = "colored_" + filename
    output_path = os.path.join(RESULTS, output_filename)

    # save BGR image
    try:
        cv2.imwrite(output_path, result_adj)
    except Exception as e:
        return f"Failed to save result: {e}"

    return render_template(
        "app.html",
        input_image=filename,
        output_image=output_filename,
        ts=int(time.time())
    )


if __name__ == "__main__":
    app.run(debug=True)
