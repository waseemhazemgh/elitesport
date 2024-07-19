from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
import io

app = Flask(__name__)

# Construct lookup tables
increase_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 75, 155, 255])(range(256))
decrease_table = UnivariateSpline(x=[0, 64, 128, 255], y=[0, 45, 95, 255])(range(256))
midtone_contrast_increase = UnivariateSpline(x=[0, 25, 51, 76, 102, 128, 153, 178, 204, 229, 255], y=[0, 13, 25, 51, 76, 128, 178, 204, 229, 242, 255])(range(256))
lowermids_increase = UnivariateSpline(x=[0, 16, 32, 48, 64, 80, 96, 111, 128, 143, 159, 175, 191, 207, 223, 239, 255], y=[0, 18, 35, 64, 81, 99, 107, 112, 121, 143, 159, 175, 191, 207, 223, 239, 255])(range(256))
uppermids_decrease = UnivariateSpline(x=[0, 16, 32, 48, 64, 80, 96, 111, 128, 143, 159, 175, 191, 207, 223, 239, 255], y=[0, 16, 32, 48, 64, 80, 96, 111, 128, 140, 148, 160, 171, 187, 216, 236, 255])(range(256))

def convert_image_to_bytes(image):
    is_success, buffer = cv2.imencode(".jpg", image)
    io_buf = io.BytesIO(buffer)
    return io_buf

def apply_warm(image):
    blue_channel, green_channel, red_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, increase_table).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, decrease_table).astype(np.uint8)
    output_image = cv2.merge((blue_channel, green_channel, red_channel))
    return output_image

def apply_cold(image):
    blue_channel, green_channel, red_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, decrease_table).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, increase_table).astype(np.uint8)
    output_image = cv2.merge((blue_channel, green_channel, red_channel))
    return output_image

def apply_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def apply_canny_edge(image):
    edges = cv2.Canny(image, 100, 200)
    return edges

def apply_blur(image):
    blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
    return blurred_image

def apply_threshold(image):
    ret, threshold_image = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    return threshold_image

def apply_skin_smoothing(image):
    smoothed_image = cv2.bilateralFilter(image, 9, 75, 75)
    return smoothed_image

def apply_softening(image):
    blurred_image = cv2.GaussianBlur(image, (0, 0), 10)
    soft_image = cv2.addWeighted(image, 1.5, blurred_image, -0.5, 0)
    return soft_image

def adjust_brightness_contrast(image, alpha=1.5, beta=10):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def enhance_skin_tone(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_shift = 10
    sat_shift = 20
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift, 0, 255)
    adjusted_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return adjusted_image

def apply_detail_enhancing(image):
    output_image = cv2.detailEnhance(image, sigma_s=15, sigma_r=0.15)
    return output_image

def apply_pencil_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    pencil_sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return pencil_sketch

def apply_warhol_style(image):
    b, g, r = cv2.split(image)
    b = b.astype(float)
    g = g.astype(float)
    r = r.astype(float)
    warhol_image = cv2.merge([b * 0.8, g, r])
    warhol_image = np.clip(warhol_image, 0, 255)
    warhol_image = warhol_image.astype(np.uint8)
    return warhol_image

def apply_gotham(image):
    blue_channel, green_channel, red_channel = cv2.split(image)
    red_channel = cv2.LUT(red_channel, midtone_contrast_increase).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, lowermids_increase).astype(np.uint8)
    blue_channel = cv2.LUT(blue_channel, uppermids_decrease).astype(np.uint8)
    output_image = cv2.merge((blue_channel, green_channel, red_channel))
    return output_image

def apply_sharpening(image):
    sharpening_kernel = np.array([[-1, -1, -1],
                                  [-1, 9.2, -1],
                                  [-1, -1, -1]])
    output_image = cv2.filter2D(src=image, ddepth=-1, kernel=sharpening_kernel)
    return output_image

def apply_invert(image):
    output_image = cv2.bitwise_not(image)
    return output_image

def apply_beauty_filter(image):
    filtered = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)
    filtered = cv2.detailEnhance(filtered, sigma_s=10, sigma_r=0.15)
    return filtered

def apply_cute_filter(image):
    alpha = 0.8
    beta = 20
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    overlay = np.full(adjusted.shape, (255, 192, 203), dtype='uint8')
    cute_image = cv2.addWeighted(adjusted, 0.7, overlay, 0.3, 0)
    return cute_image

def apply_filter(image, filter_name):
    if filter_name == "warm":
        return apply_warm(image)
    elif filter_name == "cold":
        return apply_cold(image)
    elif filter_name == "grayscale":
        return apply_grayscale(image)
    elif filter_name == "canny_edge":
        return apply_canny_edge(image)
    elif filter_name == "blur":
        return apply_blur(image)
    elif filter_name == "threshold":
        return apply_threshold(image)
    elif filter_name == "skin_smoothing":
        return apply_skin_smoothing(image)
    elif filter_name == "softening":
        return apply_softening(image)
    elif filter_name == "brightness_contrast":
        return adjust_brightness_contrast(image)
    elif filter_name == "skin_tone":
        return enhance_skin_tone(image)
    elif filter_name == "detail_enhancing":
        return apply_detail_enhancing(image)
    elif filter_name == "pencil_sketch":
        return apply_pencil_sketch(image)
    elif filter_name == "warhol_style":
        return apply_warhol_style(image)
    elif filter_name == "gotham":
        return apply_gotham(image)
    elif filter_name == "sharpening":
        return apply_sharpening(image)
    elif filter_name == "invert":
        return apply_invert(image)
    elif filter_name == "beauty_filter":
        return apply_beauty_filter(image)
    elif filter_name == "cute_filter":
        return apply_cute_filter(image)
    else:
        raise ValueError(f"Unknown filter name: {filter_name}")

def apply_filter_to_video(video_path, filter_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # Get the video's frame width, height, and frames per second (fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # Using 'mp4v' codec for .mp4 files
    out_path = 'output_video.mp4'
    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        filtered_frame = apply_filter(frame, filter_name)
        # Ensure the filtered_frame is in the correct color format if needed
        if len(filtered_frame.shape) == 2:  # if the frame is grayscale
            filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)
        out.write(filtered_frame)

    cap.release()
    out.release()
    return out_path

@app.route('/apply-warm', methods=['POST'])
def warm_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_warm(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'warm')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-cold', methods=['POST'])
def cold_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_cold(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'cold')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-grayscale', methods=['POST'])
def grayscale_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_grayscale(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'grayscale')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-canny-edge', methods=['POST'])
def canny_edge_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_canny_edge(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'canny_edge')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-blur', methods=['POST'])
def blur_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_blur(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'blur')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-threshold', methods=['POST'])
def threshold_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_threshold(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'threshold')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-skin-smoothing', methods=['POST'])
def skin_smoothing_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_skin_smoothing(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'skin_smoothing')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-softening', methods=['POST'])
def softening_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_softening(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'softening')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/adjust-brightness-contrast', methods=['POST'])
def brightness_contrast_filter():
    file = request.files['file']
    alpha = float(request.form.get('alpha', 1.5))
    beta = int(request.form.get('beta', 10))
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = adjust_brightness_contrast(image, alpha, beta)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'brightness_contrast')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/enhance-skin-tone', methods=['POST'])
def skin_tone_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = enhance_skin_tone(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'skin_tone')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-detail-enhancing', methods=['POST'])
def detail_enhancing_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_detail_enhancing(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'detail_enhancing')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-pencil-sketch', methods=['POST'])
def pencil_sketch_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_pencil_sketch(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'pencil_sketch')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-warhol-style', methods=['POST'])
def warhol_style_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_warhol_style(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'warhol_style')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-gotham', methods=['POST'])
def gotham_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_gotham(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'gotham')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-sharpening', methods=['POST'])
def sharpening_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_sharpening(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'sharpening')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-invert', methods=['POST'])
def invert_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_invert(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'invert')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-beauty-filter', methods=['POST'])
def beauty_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_beauty_filter(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'beauty_filter')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/apply-cute-filter', methods=['POST'])
def cute_filter():
    file = request.files['file']
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        output_image = apply_cute_filter(image)
        return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        input_video_path = 'input_video.mp4'
        file.save(input_video_path)
        output_video_path = apply_filter_to_video(input_video_path, 'cute_filter')
        return send_file(output_video_path, mimetype='video/mp4')
    else:
        return jsonify({"error": "Unsupported file format"}), 400

@app.route('/', methods=['GET'])
def home():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True, port=5001)
