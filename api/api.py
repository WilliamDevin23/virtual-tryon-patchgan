from flask import Flask, request, jsonify, send_file
import numpy as np
import tensorflow as tf
import io
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from try_on.try_on import generate_tryon

app = Flask(__name__)

@app.route('/', methods=['POST'])
def generate_tryon_image() :
  size = request.form.get('size')
  
  if size is None :
    return jsonify({'error': 'Please provide size value'})
  
  if size not in ['small', 'large'] :
    return jsonify({'error': 'Please provide value "small" or "large" for size'})
  
  if 'image' not in request.files or 'clothes_image' not in request.files :
    return jsonify({'error': 'Please upload both image and clothes image'}), 400
  
  image = request.files['image']
  clothes_image = request.files['clothes_image']
  
  if image.filename == '' or clothes_image.filename == '' :
    return jsonify({'error': 'Either image or clothes image is not selected'}), 400
  
  img_np = tf.image.decode_image(image.read(), channels=3).numpy()
  clothes_img_np = tf.image.decode_image(clothes_image.read(), channels=3).numpy()
  
  if size == 'small' :
    tryon_img = generate_tryon(img_np, clothes_img_np, size='small').numpy()
  else :
    tryon_img = generate_tryon(img_np, clothes_img_np, size='large').numpy()
    
  tryon_img = tf.convert_to_tensor(np.clip((tryon_img * 255).astype(np.uint8), 0, 255), dtype=tf.uint8)
  encoded_img = tf.io.encode_jpeg(tryon_img).numpy()
  
  return send_file(
    io.BytesIO(encoded_img),
    mimetype="image/jpeg",
    as_attachment=False,
    download_name="tryon.jpg"
  )
  
if __name__ == '__main__' :
  app.run(debug=True)