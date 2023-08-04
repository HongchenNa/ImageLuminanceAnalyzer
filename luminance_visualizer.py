#!/usr/bin/env python3

import cv2
import numpy as np
import scipy.interpolate as interp
import os

# Get user input
# read input from user
img_path = input("Please enter the image file path: ").strip()
print("You entered:", img_path)

# Ensure that img_path is a string
if isinstance(img_path, str):
    img = cv2.imread(img_path)  # read image
else:
    print(f"Unexpected type for img_path: {type(img_path)}")

# Separate filename and extension
filename, extension = os.path.splitext(img_path)

# Create output path
output_path = filename + '_luminance.jpg'

# Convert image to grayscale
if img is None:
    print(f"Failed to read image from {img_path}")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Map the bin indices back to the image to label the different illuminance levels
bins = np.linspace(0, 256, 12)  # we want 11 levels, so we need 12 boundaries
# subtract 1 because digitize works as 1-indexed
labels = np.digitize(gray, bins) - 1

# Scale labels to [0, 255] range and convert to 8-bit format
labels_scaled = ((labels / labels.max()) * 255).astype(np.uint8)

# Apply a color map
colored_labels = cv2.applyColorMap(labels_scaled, cv2.COLORMAP_JET)

# Create a legend image
proposed_width = int(img.shape[1] * 0.20)  # 20% of the image width
# Ensure the width is divisible by 11
legend_width = proposed_width - (proposed_width % 11)
legend_height = int(legend_width / 11)
legend_img = np.ones((legend_height, legend_width, 3),
                     dtype=np.uint8) * 255  # white background


# Define colors for each illuminance level
colors = cv2.applyColorMap(np.linspace(
    0, 255, 12).astype(np.uint8), cv2.COLORMAP_JET)

# Loop through the colors to draw rectangles and add level text
for i, color in enumerate(colors):
    # Calculate the start and end x-coordinates of the rectangle
    start_x = i * legend_width // 11
    end_x = (i + 1) * legend_width // 11

    # Draw a rectangle filled with the color
    cv2.rectangle(legend_img, (start_x, 0), (end_x, legend_height),
                  color[0].tolist(), -1)  # -1 means filled rectangle

    # Calculate the center coordinates of the rectangle
    center_x = (start_x + end_x) // 2
    center_y = legend_height // 2

    # Calculate the desired text scale (30% of the rectangle height)
    text_scale = 0.3 * legend_height / \
        cv2.getTextSize("0", cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][1]

    # Get the size of the text box
    text_size, _ = cv2.getTextSize(
        str(i), cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)

    # Calculate the bottom-left corner of the text box
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2

    # Add the level text
    cv2.putText(legend_img, str(i), (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), 2)


# Create a histogram image
hist_img_width = legend_width               # same width as the legend
hist_img_height = int(img.shape[0] * 0.15)  # 15% of the image height
# create a blank image for the histogram
hist_img = np.zeros((hist_img_height, hist_img_width, 3), dtype=np.uint8)
cv2.normalize(hist, hist, alpha=0, beta=hist_img_height,
              norm_type=cv2.NORM_MINMAX)  # normalize the histogram

# Scale the histogram to the width of the histogram image
x_old = np.arange(256)
x_new = np.linspace(0, 255, hist_img_width)
f = interp.interp1d(x_old, hist.flatten(), kind='cubic')
hist_scaled = f(x_new)

# Define a maximum value for the y-axis
# for example, use the 95th percentile as the max
max_y = np.percentile(hist_scaled, 98)

# Clip values above the max
hist_scaled = np.clip(hist_scaled, a_min=0, a_max=max_y)

# Rescale the histogram values to the image height
hist_scaled = (hist_scaled / max_y) * hist_img_height

for i in range(1, hist_img_width):
    cv2.line(hist_img, (i-1, hist_img_height - int(hist_scaled[i-1])),
             (i, hist_img_height - int(hist_scaled[i])), (255, 255, 255), 1)  # draw the histogram

# Create a blank image the same size as the original
blank_img = np.zeros_like(colored_labels)

# Draw the histogram on the blank image
blank_img[:hist_img_height, :hist_img_width] = hist_img

# Create a mask where the histogram is
mask = np.zeros_like(colored_labels)
mask[:hist_img.shape[0], :hist_img.shape[1]] = [255, 255, 255]

# Merge the histogram image and the original image using cv2.addWeighted(), but only where the mask is
mask = mask.astype(bool)  # convert to boolean mask
blank_img[mask] = cv2.addWeighted(blank_img, 1, colored_labels, 0.5, 0)[mask]

# Add the legend and histogram image to the top-right corner of the color image
combined_img = np.concatenate(
    (legend_img, blank_img[:hist_img.shape[0], :hist_img.shape[1]]), axis=0)
colored_labels[:combined_img.shape[0], -combined_img.shape[1]:] = combined_img

# Save the image
cv2.imwrite(output_path, colored_labels)
