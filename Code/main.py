import base64
import io
import cv2
import numpy as np
from flask import jsonify
from functions_framework import http
from app import run_teeth_alignment
import traceback


@http
def align_teeth(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No image in request"}), 500

    image_data = request.files['file'].read()

    # Convert the image data to a format OpenCV can work with
    np_image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    try:
        aligned_teeth_image = run_teeth_alignment(image)
    except Exception as e:
        if isinstance(e, ValueError):
            return jsonify({"error": "Please make sure that you follow the guidelines"}), 400
        else:
            error_string = traceback.format_exc()
            return jsonify({"error": f"Unexpected error produced {e}: {str(error_string)}"}), 500

    byte_arr = io.BytesIO()
    is_success, buffer = cv2.imencode(".png", aligned_teeth_image)
    if is_success:
        byte_arr.write(buffer)
        byte_arr.seek(0)
        encoded_image = base64.b64encode(byte_arr.getvalue()).decode('utf-8')
        return jsonify({"output": encoded_image}), 200
    else:
        return jsonify({"error": "Error encoding image"}), 500
