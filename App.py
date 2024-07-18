from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
import io
import dlib
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



detector = dlib.get_frontal_face_detector()
# Load the facial landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def detect_pimples(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for red color (pimples)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine masks
    mask = mask1 + mask2

    return mask

def apply_advanced_smoothing(fc, mask):
    # Apply bilateral filter to smooth the pimples
    diameter = 15  # Diameter of each pixel neighborhood that is used during filtering
    sigmaColor = 75  # Filter sigma in the color space
    sigmaSpace = 75  # Filter sigma in the coordinate space

    smoothed = cv2.bilateralFilter(fc, diameter, sigmaColor, sigmaSpace)

    # Apply the mask to the smoothed image
    smoothed_pimples = cv2.bitwise_and(smoothed, smoothed, mask=mask)

    return smoothed_pimples

def apply_beauty_filter(image):
    # Apply edge-preserving filter
    filtered = cv2.edgePreservingFilter(image, flags=1, sigma_s=60, sigma_r=0.4)

    # Apply detail enhancement filter
    filtered = cv2.detailEnhance(filtered, sigma_s=10, sigma_r=0.15)

    return filtered

def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        raise Exception("Error: Unable to load the image")

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray, 1)  # The '1' here means to upscale the image 1 time.

    # Create a copy of the original image to apply smoothing effects
    smoothed_img = img.copy()

    # Process each detected face
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

        # Create a mask for the entire face region
        mask = np.zeros_like(gray)
        points = np.concatenate([
            landmarks[0:17],  # Jawline
            landmarks[17:22],  # Left eyebrow
            landmarks[22:27],  # Right eyebrow
            landmarks[27:36],  # Nose bridge and bottom
            landmarks[36:48],  # Eyes and area around
            landmarks[48:68]   # Mouth and area around
        ])
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)

        # Extract the face region
        face_region = cv2.bitwise_and(smoothed_img, smoothed_img, mask=mask)

        # Detect pimples in the face region
        pimple_mask = detect_pimples(face_region)

        # Apply advanced smoothing to the pimples
        smoothed_pimples = apply_advanced_smoothing(face_region, pimple_mask)

        # Blend the smoothed pimples with the original image
        mask_inv = cv2.bitwise_not(pimple_mask)
        img_bg = cv2.bitwise_and(smoothed_img, smoothed_img, mask=mask_inv)
        smoothed_img = cv2.add(img_bg, smoothed_pimples)

    # Apply beauty filter to the entire image
    beauty_img = apply_beauty_filter(smoothed_img)

    return img, beauty_img

def convert_image_to_bytes(image):
    is_success, buffer = cv2.imencode(".jpg", image)
    io_buf = io.BytesIO(buffer)
    return io_buf

@app.route('/process-image', methods=['POST'])
def process_image_route():
    file = request.files['image']
    image_path = 'uploaded_image.jpg'
    file.save(image_path)

    try:
        original_img, modified_img = process_image(image_path)
        original_img_bytes = convert_image_to_bytes(original_img)
        modified_img_bytes = convert_image_to_bytes(modified_img)

        return jsonify(
            original_image=send_file(original_img_bytes, mimetype='image/jpeg'),
            modified_image=send_file(modified_img_bytes, mimetype='image/jpeg')
        )
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/apply-warm', methods=['POST'])
def warm_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_warm(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/apply-cold', methods=['POST'])
def cold_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_cold(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/apply-grayscale', methods=['POST'])
def grayscale_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_grayscale(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/apply-canny-edge', methods=['POST'])
def canny_edge_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_canny_edge(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/apply-blur', methods=['POST'])
def blur_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_blur(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/apply-threshold', methods=['POST'])
def threshold_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_threshold(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/apply-skin-smoothing', methods=['POST'])
def skin_smoothing_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_skin_smoothing(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/apply-softening', methods=['POST'])
def softening_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_softening(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/adjust-brightness-contrast', methods=['POST'])
def brightness_contrast_filter():
    file = request.files['image']
    alpha = float(request.form.get('alpha', 1.5))
    beta = int(request.form.get('beta', 10))
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = adjust_brightness_contrast(image, alpha, beta)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/enhance-skin-tone', methods=['POST'])
def skin_tone_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = enhance_skin_tone(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/apply-detail-enhancing', methods=['POST'])
def detail_enhancing_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_detail_enhancing(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/apply-pencil-sketch', methods=['POST'])
def pencil_sketch_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_pencil_sketch(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/apply-warhol-style', methods=['POST'])
def warhol_style_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_warhol_style(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/apply-gotham', methods=['POST'])
def gotham_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_gotham(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/apply-sharpening', methods=['POST'])
def sharpening_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_sharpening(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')

@app.route('/apply-invert', methods=['POST'])
def invert_filter():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    output_image = apply_invert(image)
    return send_file(convert_image_to_bytes(output_image), mimetype='image/jpeg')


@app.route('/', methods=['GET'])
def home():
    return "Hello, World!"


if __name__ == '__main__':
    app.run(debug=True, port=5000)



@app.route('/', methods=['GET'])
def home():
    return "Hello, World!"