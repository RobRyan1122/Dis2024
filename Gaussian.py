import os
import cv2
import numpy as np

# Define input and output directories
input_folder = "GaussianTarget"
output_folder = "GaussianOutput"

# Create the output directory if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to apply Gaussian filter to an image
def apply_gaussian_filter(image, kernel_size=(5, 5), sigma=1):
    return cv2.GaussianBlur(image, kernel_size, sigma)

# Get standard deviation from user input
try:
    sigma = float(input("Enter the standard deviation for the Gaussian filter: "))
except ValueError:
    print("Invalid input. Using default standard deviation of 1.")
    sigma = 1

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # Check if the file is an image (by extension)
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error reading file: {file_path}")
            continue

        # Apply Gaussian filter
        filtered_image = apply_gaussian_filter(image, sigma=sigma)

        # Save the processed image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, filtered_image)
        print(f"Processed and saved: {output_path}")

print("Processing complete.")
