import json
import os
import uuid

import cv2
import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request, send_file
from fpdf import FPDF, XPos, YPos
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from werkzeug.utils import secure_filename

from config.config import ALLOWED_EXTENSIONS, DEBUG, SECRET_KEY, UPLOAD_FOLDER
from src.features.inference import colab_style_inference, load_config, load_model, preprocess_image
from src.utils.input_stream import allowed_file

app = Flask(__name__, template_folder = 'templates')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS
app.config['DEBUG'] = DEBUG
app.config['SECRET_KEY'] = SECRET_KEY

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def create_unique_filename(filename):
    name, ext = os.path.splitext(filename)
    unique_id = uuid.uuid4().hex
    return f"{unique_id}{ext}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    error_message = None
    files_info = []
    if request.method == 'POST':
        if 'file' not in request.files:
            error_message = "No file part in the request."
            return render_template('upload.html', error_message = error_message)

        files = request.files.getlist('file')
        if len(files) == 0:
            error_message = "No files selected for uploading."
            return render_template('upload.html', error_message = error_message)

        for file in files:
            if file and allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
                original_filename = secure_filename(file.filename)
                file_extension = os.path.splitext(original_filename)[1].lower()
                upload_subfolder = os.path.join(app.config['UPLOAD_FOLDER'], file_extension[1:])
                os.makedirs(upload_subfolder, exist_ok = True)
                unique_filename = create_unique_filename(original_filename)
                file_path = os.path.join(upload_subfolder, unique_filename)
                file.save(file_path)

                try:
                    probs = colab_style_inference(file_path, model, class_names)
                    predictions = {name: float(prob) for name, prob in zip(class_names, probs) if prob >= 0.1}
                    if not predictions:
                        top_idx = np.argmax(probs)
                        predictions[class_names[top_idx]] = float(probs[top_idx])
                    top_prediction = max(predictions, key = predictions.get)

                    img = np.array(Image.open(file_path).convert('RGB'))
                    img_resized = cv2.resize(img, tuple(config['dataset']['img_size']))
                    img_resized = np.float32(img_resized) / 255
                    input_tensor = preprocess_image(file_path, tuple(config['dataset']['img_size']),
                                                    config['dataset']['mean'], config['dataset']['std'])
                    target_class_idx = class_names.index(top_prediction)
                    targets = [ClassifierOutputTarget(target_class_idx)]
                    target_layers = [model.layer4[-1]]
                    model.eval()
                    with torch.set_grad_enabled(True):
                        with GradCAM(model = model, target_layers = target_layers) as cam:
                            grayscale_cams = cam(input_tensor = input_tensor, targets = targets)
                            cam_image = show_cam_on_image(img_resized, grayscale_cams[0, :], use_rgb = True)
                    gradcam_filename = f"gradcam_{unique_filename}"
                    gradcam_output_path = os.path.join(upload_subfolder, gradcam_filename)
                    Image.fromarray(cam_image).save(gradcam_output_path)

                    files_info.append({
                        'original_image': unique_filename,
                        'gradcam_image': gradcam_filename,
                        'prediction': top_prediction,
                        'all_predictions': predictions,
                        'upload_subfolder': file_extension[1:]
                    })

                except Exception as e:
                    print(f"Error processing {original_filename}: {e}")
                    files_info.append({
                        'original_image': original_filename,
                        'error': str(e)
                    })

    return render_template('upload.html', files_info = files_info, error_message = error_message)


@app.route('/generate_pdf', methods = ['POST'])
def generate_pdf():
    files_info = json.loads(request.form['files_info'])

    # Create a landscape A4 PDF
    pdf = FPDF(orientation = 'L', unit = 'mm', format = 'A4')
    pdf.set_auto_page_break(auto = True, margin = 15)

    # Add a cover page
    pdf.add_page()
    pdf.set_font("Helvetica", size = 24, style = 'B')
    pdf.set_text_color(0, 137, 123)  # Use teal color for title
    pdf.cell(0, 20, "Lung X-Ray Analysis Report", align = 'C', new_x = XPos.LMARGIN, new_y = YPos.NEXT)

    # Add date
    pdf.set_font("Helvetica", size = 12)
    pdf.set_text_color(80, 80, 80)
    from datetime import datetime
    today = datetime.now().strftime("%B %d, %Y")
    pdf.cell(0, 10, f"Generated on: {today}", align = 'C', new_x = XPos.LMARGIN, new_y = YPos.NEXT)

    # Add logo/icon
    pdf.image(os.path.join('static', 'images/lungs.png'), x = 130, y = 70, w = 40)
    pdf.set_font("Helvetica", size = 12)
    pdf.set_text_color(100, 100, 100)
    pdf.set_y(160)
    pdf.cell(0, 10, "LungScan AI - Advanced Chest X-Ray Analysis", align = 'C', new_x = XPos.LMARGIN, new_y = YPos.NEXT)
    pdf.cell(0, 10, "Powered by Artificial Intelligence", align = 'C', new_x = XPos.LMARGIN, new_y = YPos.NEXT)

    # Add summary information
    pdf.add_page()
    pdf.set_font("Helvetica", size = 16, style = 'B')
    pdf.set_text_color(0, 137, 123)  # Use teal color for headings
    pdf.cell(0, 10, "Analysis Summary", align = 'L', new_x = XPos.LMARGIN, new_y = YPos.NEXT)

    pdf.set_font("Helvetica", size = 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Total images analyzed: {len(files_info)}", align = 'L', new_x = XPos.LMARGIN, new_y = YPos.NEXT)

    # Count normal vs abnormal findings
    normal_count = sum(1 for file in files_info if file['prediction'] == 'No Finding')
    abnormal_count = len(files_info) - normal_count
    pdf.cell(0, 8, f"Normal findings: {normal_count}", align = 'L', new_x = XPos.LMARGIN, new_y = YPos.NEXT)
    pdf.cell(0, 8, f"Abnormal findings: {abnormal_count}", align = 'L', new_x = XPos.LMARGIN, new_y = YPos.NEXT)

    # Add table of contents
    pdf.ln(10)
    pdf.set_font("Helvetica", size = 14, style = 'B')
    pdf.set_text_color(0, 137, 123)
    pdf.cell(0, 10, "Contents", align = 'L', new_x = XPos.LMARGIN, new_y = YPos.NEXT)

    pdf.set_font("Helvetica", size = 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "1. Analysis Summary", align = 'L', new_x = XPos.LMARGIN, new_y = YPos.NEXT)
    pdf.cell(0, 8, "2. Detailed Results", align = 'L', new_x = XPos.LMARGIN, new_y = YPos.NEXT)
    pdf.cell(0, 8, "3. Interpretation Guide", align = 'L', new_x = XPos.LMARGIN, new_y = YPos.NEXT)

    # Add results page
    pdf.add_page()
    pdf.set_font("Helvetica", size = 16, style = 'B')
    pdf.set_text_color(0, 137, 123)
    pdf.cell(0, 10, "Detailed Results", align = 'L', new_x = XPos.LMARGIN, new_y = YPos.NEXT)
    pdf.ln(5)

    # Define table settings
    pdf.set_font("Helvetica", size = 10)
    col_width = [15, 50, 50, 70, 50]  # Adjusted column widths
    image_size = 40
    line_height = 8

    # Table header
    pdf.set_fill_color(0, 137, 123)  # Teal header background
    pdf.set_text_color(255, 255, 255)  # White text
    pdf.set_font("Helvetica", style = 'B', size = 10)
    headers = ["No.", "Original Image", "Grad-CAM Analysis", "Findings", "Confidence"]

    for i, header in enumerate(headers):
        if i < len(headers) - 1:
            pdf.cell(col_width[i], line_height, header, border = 1, align = 'C', fill = True)
        else:
            pdf.cell(col_width[i], line_height, header, border = 1, align = 'C', fill = True, new_x = XPos.LMARGIN,
                     new_y = YPos.NEXT)

    # Table data
    pdf.set_fill_color(240, 248, 255)  # Light blue alternate row color
    pdf.set_text_color(0, 0, 0)
    fill = False

    for idx, file_info in enumerate(files_info, start = 1):
        y_start = pdf.get_y()
        max_height = image_size

        # Column 1: Index number
        pdf.cell(col_width[0], image_size, str(idx), border = 1, align = 'C', fill = fill)

        # Column 2: Original Image
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file_info['upload_subfolder'],
                                           file_info['original_image'])
        try:
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.cell(col_width[1], image_size, '', border = 1, fill = fill)
            pdf.image(original_image_path, x = x + (col_width[1] - image_size) / 2, y = y + 2, w = image_size - 4,
                      h = image_size - 4)
        except Exception as e:
            pdf.cell(col_width[1], image_size, f"Error loading image", border = 1, align = 'C', fill = fill)

        # Column 3: Grad-CAM Image
        gradcam_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file_info['upload_subfolder'],
                                          file_info['gradcam_image'])
        try:
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.cell(col_width[2], image_size, '', border = 1, fill = fill)
            pdf.image(gradcam_image_path, x = x + (col_width[2] - image_size) / 2, y = y + 2, w = image_size - 4,
                      h = image_size - 4)
        except Exception as e:
            pdf.cell(col_width[2], image_size, f"Error loading image", border = 1, align = 'C', fill = fill)

        # Column 4: Findings
        x = pdf.get_x()
        y = pdf.get_y()
        pdf.multi_cell(col_width[3], line_height,
                       f"Primary Finding: {file_info['prediction']}\n\n" +
                       "File: " + file_info['original_image'] + "\n\n" +
                       "Notes: Areas highlighted in the Grad-CAM image indicate regions that significantly influenced the AI's diagnosis.",
                       border = 1, align = 'L', fill = fill)

        # Calculate how much vertical space was used
        height_used = pdf.get_y() - y
        if height_used > max_height:
            max_height = height_used

        # Reset position for next column
        pdf.set_xy(x + col_width[3], y)

        # Column 5: Confidence
        if 'all_predictions' in file_info:
            confidence_text = ""
            for condition, prob in file_info['all_predictions'].items():
                percentage = round(prob * 100, 1)
                confidence_text += f"{condition}: {percentage}%\n"
            pdf.multi_cell(col_width[4], line_height, confidence_text, border = 1, align = 'L', fill = fill)
        else:
            pdf.cell(col_width[4], image_size, "N/A", border = 1, align = 'C', fill = fill)

        # Move to next row, ensuring all columns are of equal height
        pdf.set_y(y_start + max_height)
        fill = not fill

        # Add page break if we're close to the bottom
        if pdf.get_y() > pdf.h - 30:
            pdf.add_page()
            # Repeat the header on new page
            pdf.set_fill_color(0, 137, 123)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", style = 'B', size = 10)
            for i, header in enumerate(headers):
                if i < len(headers) - 1:
                    pdf.cell(col_width[i], line_height, header, border = 1, align = 'C', fill = True)
                else:
                    pdf.cell(col_width[i], line_height, header, border = 1, align = 'C', fill = True,
                             new_x = XPos.LMARGIN, new_y = YPos.NEXT)
            pdf.ln(line_height)
            pdf.set_fill_color(240, 248, 255)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", size = 10)

    # Add interpretation guide
    pdf.add_page()
    pdf.set_font("Helvetica", size = 16, style = 'B')
    pdf.set_text_color(0, 137, 123)
    pdf.cell(0, 10, "Interpretation Guide", align = 'L', new_x = XPos.LMARGIN, new_y = YPos.NEXT)

    pdf.set_font("Helvetica", size = 11)
    pdf.set_text_color(80, 80, 80)

    # Add guide content
    guide_text = """
This report is generated using AI-based analysis of chest X-ray images. The following information will help you interpret the results:

1. Original Image: The unmodified chest X-ray that was uploaded for analysis.
2. Grad-CAM Analysis: This visualization highlights areas of the X-ray that were most influential in the AI's diagnosis. Warmer colors (red, yellow) indicate regions of high importance for the predicted condition.
3. Findings: The primary diagnosis determined by the AI system, along with the filename of the analyzed image.
4. Confidence: The confidence levels (as percentages) for various conditions detected in the image.

Important Notes:
- This AI analysis is intended to assist medical professionals and should not replace clinical judgment.
- The Grad-CAM visualization helps explain the AI's decision-making process by highlighting areas of interest.
- Multiple conditions may be detected in a single image with varying confidence levels.
"""
    pdf.multi_cell(0, 6, guide_text, align = 'L')

    # Add footer
    pdf.set_auto_page_break(auto = False)
    for page in range(1, pdf.page_no() + 1):
        pdf.page = page
        pdf.set_y(-15)
        pdf.set_font("Helvetica", size = 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, f"Page {page} of {pdf.page_no()}", align = 'C')
        pdf.cell(0, 10, "Generated by LungScan AI", align = 'R', new_x = XPos.RIGHT, new_y = YPos.TOP)

    # Save the PDF
    pdf_output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'LungScan_AI_Report.pdf')
    pdf.output(pdf_output_path)

    return send_file(pdf_output_path, as_attachment = True,
                     download_name = f"LungScan_AI_Report_{today.replace(' ', '_')}.pdf")


if __name__ == '__main__':
    try:
        config_path = 'config/train-config.yaml'
        config = load_config(config_path)
        model_path = os.path.join(config['model']['exp_folder'], config['model']['model_name'])
        model, loaded_class_names = load_model(config['model'], model_path)
        class_names = loaded_class_names if loaded_class_names else [
            'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
            'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
            'Pleural_thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia',
            'No Finding'
        ]
        print("Class names (length: {}):".format(len(class_names)))
        for i, name in enumerate(class_names):
            print(f"{i}: {name}")
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            test_output = model(test_input)
            print(f"Model test output shape: {test_output.shape}")
            print("Sample outputs:", torch.sigmoid(test_output[0]))
        print("Model loaded successfully with classes:", class_names)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")

    app.run(debug = DEBUG)
