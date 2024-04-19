from PIL import Image
import os

def convert_images(input_directory, output_directory, target_format='jpg'):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = os.listdir(input_directory)

    for file in files:
        if file.lower().endswith('.jpg') or file.lower().endswith('.png'):
            input_path = os.path.join(input_directory, file)
            img = Image.open(input_path)
            output_path = os.path.join(output_directory, file)
            img.save(output_path)   

if __name__ == "__main__":
    input_directory = 'ssl_datasets/'
    output_directory = 'ssl_datasets_preprocessed/'
    # Specify the target format ('jpg' or 'png')
    target_format = 'png'
    convert_images(input_directory, output_directory, target_format)
