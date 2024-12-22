import cv2
import os
import Vars
import containsanimal as ca
from tensorflow.keras.applications import ResNet50, VGG16, InceptionV3, Xception, MobileNet, DenseNet121
import csv
import datetime

#def display_first_10_images(image_dir='COD10K-v3/train/image'):
def display_first_10_images(image_dir=r'C:\Users\PC\PycharmProjects\dis\textest3'):

    # Get a list of files in the directory
    image_files = sorted(os.listdir(image_dir))
    # Set number of tests
    Vars.num_of_img = int(input("How many images do you want to test?: "))

    # Select the first N image files based on user input
    selected_images = image_files[:Vars.num_of_img]

    # Ask user which model to use
    print("Select the model you want to use:")
    print("1. ResNet50")
    print("2. VGG16")
    print("3. InceptionV3")

    
    print("4. Xception")
    print("5. MobileNet")
    print("6. DenseNet121")
    model_choice = int(input("Enter the number of the model (1-6): "))

    # Load the chosen model
    model_name_map = {
        1: ('resnet50', ResNet50),
        2: ('vgg16', VGG16),
        3: ('inceptionv3', InceptionV3),
        4: ('xception', Xception),
        5: ('mobilenet', MobileNet),
        6: ('densenet121', DenseNet121)
    }

    model_name, model_class = model_name_map[model_choice]
    print(f"Using {model_name} model...")
    model = model_class(weights='imagenet')

    # List to collect results
    results = []

    # Read and display each image
    for i, image_file in enumerate(selected_images):
        # Construct the full path to the image file
        image_path = os.path.join(image_dir, image_file)

        # Read the image using OpenCV
        img = cv2.imread(image_path)

        # Display the image in an individual window
       # window_name = f'Image {i + 1}'
       # cv2.imshow(window_name, img)

        # Predict if the image contains an animal and get the decoded predictions
        decoded_predictions, contains_animal_flag = ca.predict_image(model, image_path, model_name)

        print(f"Decoded predictions for {image_file}: {decoded_predictions}")
        print(f"Contains animal flag for {image_file}: {contains_animal_flag}")

        # Output if the image contains an animal
        if contains_animal_flag:
            print(f"The image {image_file} contains an animal.")
        else:
            print(f"The image {image_file} does not contain an animal.")

        # Collect result for CSV
        results.append({
            'image': image_file,
            'top_prediction': decoded_predictions[0][0][1],
            'top_probability': decoded_predictions[0][0][2],
            'second_prediction': decoded_predictions[0][1][1],
            'second_probability': decoded_predictions[0][1][2],
            'third_prediction': decoded_predictions[0][2][1],
            'third_probability': decoded_predictions[0][2][2],
            'contains_animal': contains_animal_flag
        })

        # Wait for a key press to close the window
       # cv2.waitKey(0)
       # cv2.destroyWindow(window_name)

    # Destroy all windows (if any remain open)
    cv2.destroyAllWindows()

    # Ensure CSVarchive directory exists
    csv_dir = 'CSVarchive'
    os.makedirs(csv_dir, exist_ok=True)

    # Generate CSV file name
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_name = f"{csv_dir}/prediction_results_{model_name}_{current_time}.csv"

    # Write results to CSV
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Top Prediction', 'Top Probability', 'Second Prediction', 'Second Probability', 'Third Prediction', 'Third Probability', 'Contains Animal'])

        for result in results:
            writer.writerow([
                result['image'],
                result['top_prediction'],
                f"{result['top_probability']:.2f}",
                result['second_prediction'],
                f"{result['second_probability']:.2f}",
                result['third_prediction'],
                f"{result['third_probability']:.2f}",
                result['contains_animal']
            ])

    print(f"Results have been saved to {csv_file_name}")

if __name__ == '__main__':
    display_first_10_images()
