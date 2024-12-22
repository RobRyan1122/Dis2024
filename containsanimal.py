import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess, decode_predictions as resnet_decode
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess, decode_predictions as vgg16_decode
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionv3_preprocess, decode_predictions as inceptionv3_decode
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess, decode_predictions as xception_decode
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess, decode_predictions as mobilenet_decode
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess, decode_predictions as densenet_decode
import numpy as np

# Mapping for preprocessing and decoding, including target sizes
model_preprocess_map = {
    'resnet50': (resnet_preprocess, resnet_decode, (224, 224)),
    'vgg16': (vgg16_preprocess, vgg16_decode, (224, 224)),
    'inceptionv3': (inceptionv3_preprocess, inceptionv3_decode, (299, 299)),
    'xception': (xception_preprocess, xception_decode, (299, 299)),
    'mobilenet': (mobilenet_preprocess, mobilenet_decode, (224, 224)),
    'densenet121': (densenet_preprocess, densenet_decode, (224, 224))
}

def load_and_preprocess_image(img_path, model_name):
    _, _, target_size = model_preprocess_map[model_name]
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preprocess_input, _, _ = model_preprocess_map[model_name]
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(model, img_path, model_name):
    img_array = load_and_preprocess_image(img_path, model_name)
    predictions = model.predict(img_array)
    _, decode_predictions, _ = model_preprocess_map[model_name]
    decoded_predictions = decode_predictions(predictions, top=3)

    contains_animal_flag = contains_animal(decoded_predictions)
    return decoded_predictions, contains_animal_flag

def contains_animal(decoded_predictions):
    animal_names = [
        'dog', 'cat', 'horse', 'elephant', 'bear', 'zebra', 'giraffe',
        'cow', 'sheep', 'goat', 'rhinoceros', 'hippopotamus', 'kangaroo',
        'penguin', 'eagle', 'owl', 'duck', 'frog', 'turtle', 'snake',
        'crab', 'lobster', 'octopus'
    ]

    for _, class_name, _ in decoded_predictions[0]:
        if class_name in animal_names:
            print(f"Animal detected: {class_name}")
            return True
    print("No animal detected")
    return False

if __name__ == '__main__':
    pass
