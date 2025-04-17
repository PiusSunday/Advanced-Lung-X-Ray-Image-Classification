from PIL import Image

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def validate_image_aspect_ratio(file):
    image = Image.open(file)
    width, height = image.size
    return width == height