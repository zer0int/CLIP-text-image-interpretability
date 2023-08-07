# Let's write a Python script that takes an input folder with images, writes the filename of each image into the image as text with draw.text, and overwrites the original files.

import os
import glob
from PIL import Image, ImageDraw, ImageFont

# Specify the directory that contains the heatmap images
directory = 'VIS'

# Get a list of all image files in the directory
image_files = glob.glob(os.path.join(directory, '*.png'))

def get_font_in_order(font_names, font_size):
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, font_size)
        except IOError:
            continue
    raise ValueError(f"None of the fonts {font_names} are available.")


# Go through all images
for image_file in image_files:
    img = Image.open(image_file).convert('RGBA')
    draw = ImageDraw.Draw(img)

    # Choose your font and size
    font_names = ["arialn.ttf", "DejaVuSansCondensed.ttf", "segoeui.ttf", "NotoSans-Regular.ttf", "symbola.ttf", "arial.ttf"]
    font_size = 15
    font = get_font_in_order(font_names, font_size)

    # Write the filename into the image
    draw.text((10, 10), os.path.basename(image_file), fill='white', font=font)

    # Save the image, overwriting the original file
    img.save(image_file)

print('\n\n     3. Done writing filename overlay into images.')

