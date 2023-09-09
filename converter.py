import os
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

test_folder = os.path.join(os.getcwd(), 'datasets/Camera not Present')
dest_folder = os.path.join(os.getcwd(), 'datasets/no_camera_present')

for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)

    image = Image.open(img_path)
    # Convert to JPEG
    new_path = os.path.join(dest_folder, img_name.split(".")[0] + ".jpg")
    image.convert('RGB').save(new_path)
    print(new_path)
