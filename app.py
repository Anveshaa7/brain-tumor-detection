from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import io
import base64
from PIL import Image
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import os

app = Flask(__name__)
CORS(app)

# Load VGG19 model
MODEL_PATH = 'models/fine_tuned_vgg19_brain_tumor.h5'
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Class names
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = (224, 224)

# GradCAM implementation
class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        
        try:
            target_layer = model.get_layer(self.layer_name)
        except:
            for layer in model.layers:
                if hasattr(layer, 'layers'):
                    try:
                        target_layer = layer.get_layer(self.layer_name)
                        break
                    except:
                        continue
        
        self.grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output]
        )
    
    def _find_target_layer(self):
        for layer in reversed(self.model.layers):
            try:
                if len(layer.output_shape) == 4:
                    return layer.name
            except:
                if hasattr(layer, 'layers'):
                    for sublayer in reversed(layer.layers):
                        try:
                            if len(sublayer.output_shape) == 4:
                                return sublayer.name
                        except:
                            continue
        return 'block5_conv4'
    
    def generate_heatmap(self, img_array, pred_index=None):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        
        conv_outputs = conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, :]
        
        heatmap = tf.reduce_mean(conv_outputs, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, heatmap, original_img, alpha=0.4):
        heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        if original_img.max() <= 1.0:
            original_img = np.uint8(255 * original_img)
        
        original_img = np.array(original_img, dtype=np.uint8)
        heatmap = np.array(heatmap, dtype=np.uint8)
        
        superimposed_img = cv2.addWeighted(heatmap, alpha, original_img, 1-alpha, 0)
        
        return superimposed_img

# Initialize GradCAM
gradcam = GradCAM(model)

def preprocess_image(img):
    """Preprocess image for model prediction"""
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def image_to_base64(img):
    """Convert image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle MRI image prediction"""
    try:
        start_time = datetime.now()
        
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Load and preprocess image
        img = Image.open(file.stream).convert('RGB')
        img_array = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        pred_class_idx = np.argmax(predictions[0])
        pred_class_name = CLASS_NAMES[pred_class_idx]
        confidence = float(predictions[0][pred_class_idx] * 100)
        
        # Generate GradCAM heatmap
        heatmap = gradcam.generate_heatmap(img_array, pred_index=pred_class_idx)
        
        # Create overlay
        original_img_array = image.img_to_array(img.resize(IMG_SIZE))
        superimposed_img = gradcam.overlay_heatmap(heatmap, original_img_array)
        
        # Convert to PIL and base64
        heatmap_pil = Image.fromarray(superimposed_img.astype('uint8'))
        heatmap_base64 = image_to_base64(heatmap_pil)
        original_base64 = image_to_base64(img.resize(IMG_SIZE))
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'class': pred_class_name.capitalize(),
                'confidence': round(confidence, 2),
                'all_probabilities': {
                    CLASS_NAMES[i].capitalize(): round(float(predictions[0][i] * 100), 2)
                    for i in range(len(CLASS_NAMES))
                }
            },
            'images': {
                'original': original_base64,
                'heatmap': heatmap_base64
            },
            'metadata': {
                'model': 'VGG19',
                'model_accuracy': '89.71%',
                'processing_time': round(processing_time, 2),
                'image_size': f'{IMG_SIZE[0]}x{IMG_SIZE[1]}',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-report', methods=['POST'])
def download_report():
    """Generate and download PDF report"""
    try:
        data = request.json
        
        # Create PDF
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 24)
        c.drawString(100, height - 80, "Brain Tumor Detection Report")
        
        # Divider
        c.line(100, height - 100, width - 100, height - 100)
        
        # Patient/Report Info
        c.setFont("Helvetica", 12)
        y = height - 140
        c.drawString(100, y, f"Report Generated: {data['metadata']['timestamp']}")
        y -= 25
        c.drawString(100, y, f"Model Used: {data['metadata']['model']}")
        y -= 25
        c.drawString(100, y, f"Model Accuracy: {data['metadata']['model_accuracy']}")
        y -= 25
        c.drawString(100, y, f"Processing Time: {data['metadata']['processing_time']}s")
        
        # Diagnosis Results
        y -= 50
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, y, "Diagnosis Results:")
        
        y -= 30
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y, f"Detected: {data['prediction']['class']}")
        
        y -= 25
        c.setFont("Helvetica", 12)
        c.drawString(100, y, f"Confidence: {data['prediction']['confidence']}%")
        
        # All Probabilities
        y -= 40
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, y, "Classification Probabilities:")
        
        y -= 25
        c.setFont("Helvetica", 11)
        for class_name, prob in data['prediction']['all_probabilities'].items():
            c.drawString(120, y, f"• {class_name}: {prob}%")
            y -= 20
        
        # Disclaimer
        y -= 40
        c.setFont("Helvetica-Bold", 10)
        c.setFillColorRGB(0.8, 0, 0)
        c.drawString(100, y, "IMPORTANT DISCLAIMER:")
        y -= 20
        c.setFont("Helvetica", 9)
        c.setFillColorRGB(0, 0, 0)
        disclaimer_text = [
            "This is an AI-powered diagnostic aid tool and should not be used as the sole",
            "basis for medical decisions. Always consult qualified healthcare professionals",
            "for proper diagnosis and treatment. This report is for research and educational",
            "purposes only."
        ]
        for line in disclaimer_text:
            c.drawString(100, y, line)
            y -= 15
        
        # Footer
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(100, 50, "Brain Tumor Detection System | Powered by VGG19 Deep Learning Model")
        c.drawString(100, 35, "© 2024 Medical AI Research Project")
        
        c.save()
        buffer.seek(0)
        
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'brain_tumor_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("="*70)
    print("🧠 Brain Tumor Detection System - Starting Server...")
    print("="*70)
    print(f"Model: VGG19 (Accuracy: 89.71%)")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Classes: {CLASS_NAMES}")
    print("="*70)
    print("🌐 Server running at: http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)