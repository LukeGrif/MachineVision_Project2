import numpy as np
from skimage.color import rgb2hsv
from scipy.ndimage import label, binary_closing, binary_opening
from skimage.transform import resize
import matplotlib.pyplot as plt

def propose_regions(rgb_image):
    hsv_image = rgb2hsv(rgb_image)

    # Improved Hue wrap-around logic
    hue = hsv_image[..., 0]
    sat = hsv_image[..., 1]
    val = hsv_image[..., 2]

    hue_mask = (hue > 0.90) | (hue < 0.05)
    sat_mask = sat > 0.3
    val_mask = val > 0.3

    combined_mask = hue_mask & sat_mask & val_mask

    # Morphological operations (already good)
    cleaned_mask = binary_closing(combined_mask, structure=np.ones((5, 5)))
    cleaned_mask = binary_opening(cleaned_mask, structure=np.ones((3, 3)))

    labeled_mask, num_features = label(cleaned_mask)

    regions = []
    for i in range(1, num_features + 1):
        rows, cols = np.where(labeled_mask == i)
        if len(rows) == 0:
            continue
        y1, y2 = np.min(rows), np.max(rows)
        x1, x2 = np.min(cols), np.max(cols)

        width, height = x2 - x1, y2 - y1
        aspect_ratio = width / height if height > 0 else 0

        if (width * height > 100 and
            0.5 < aspect_ratio < 2.0 and
            width > 20 and height > 20):
            regions.append((x1, y1, x2, y2))


    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Binary Mask (Red Detection)")
    plt.imshow(hue_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Cleaned Mask")
    plt.imshow(sat_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Cleaned Mask")
    plt.imshow(val_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Binary Mask (Red Detection)")
    plt.imshow(combined_mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Cleaned Mask")
    plt.imshow(cleaned_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return regions
