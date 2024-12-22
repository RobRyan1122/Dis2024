import os
import numpy as np
import cv2


input_folder = 'Fourier target'
output_folder = 'Fourier output'


os.makedirs(output_folder, exist_ok=True)

# normalize processed image to match the original
def normalize_image(processed_image, original_image):
    original_min, original_max = original_image.min(), original_image.max()
    processed_min, processed_max = processed_image.min(), processed_image.max()

    # Scale the processed image to the original range
    normalized_image = (processed_image - processed_min) / (processed_max - processed_min)
    normalized_image = normalized_image * (original_max - original_min) + original_min
    return np.uint8(np.clip(normalized_image, 0, 255))

def enhance_low_frequencies_channel(channel, enhancement_factor):
    # Fourier Transform
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = np.abs(fshift)

    # Increase low frequencies by enhancement factor
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    low_freq_radius = min(rows, cols) // 4  # Define the radius for low frequencies

    # Isolate low frequencies
    mask = np.zeros_like(channel, dtype=np.uint8)
    cv2.circle(mask, (ccol, crow), low_freq_radius, 1, -1)

    # Enhance the low frequencies
    fshift_real = np.real(fshift)
    fshift_imag = np.imag(fshift)
    fshift_real[mask == 1] *= enhancement_factor
    fshift_imag[mask == 1] *= enhancement_factor

    fshift = fshift_real + 1j * fshift_imag

    #inverse Fourier Transform
    f_ishift = np.fft.ifftshift(fshift)
    channel_back = np.fft.ifft2(f_ishift)
    channel_back = np.abs(channel_back)

    # Normalize the processed channel
    channel_back = normalize_image(channel_back, channel)

    return channel_back

# Ask the user for the enhancement factor
enhancement_factor = float(input("Enter the enhancement factor for low frequencies (e.g., 1.5): "))

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Read the image
        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)

        # Split the image into color channels
        b, g, r = cv2.split(image)

        # Apply the Fourier transform and low-frequency enhancement to each channel
        b_processed = enhance_low_frequencies_channel(b, enhancement_factor)
        g_processed = enhance_low_frequencies_channel(g, enhancement_factor)
        r_processed = enhance_low_frequencies_channel(r, enhancement_factor)

        # Merge the channels back into a color image
        processed_image = cv2.merge((b_processed, g_processed, r_processed))

        # Save the processed image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, processed_image)
        print(f"Processed and saved: {output_path}")

print("Processing complete. Check the 'Fourier output' folder for results.")
