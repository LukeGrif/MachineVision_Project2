import numpy as np
from skimage.color import rgb2hsv
from scipy.ndimage import label, binary_closing, binary_opening
from skimage.transform import resize


def propose_regions(rgb_image):
    """
    Detect potential speed sign regions in an RGB image.
    Returns list of bounding boxes (x1, y1, x2, y2).
    """
    # Convert to HSV
    hsv_image = rgb2hsv(rgb_image)

    # Threshold for red rims (Hue around 0 with reasonable S/V)
    hue_mask = (hsv_image[..., 0] > 0.95) | (hsv_image[..., 0] < 0.05)
    sat_mask = hsv_image[..., 1] > 0.5
    val_mask = hsv_image[..., 2] > 0.3

    combined_mask = hue_mask & sat_mask & val_mask

    # Morphological operations to clean up the mask
    cleaned_mask = binary_closing(combined_mask, structure=np.ones((5, 5)))
    cleaned_mask = binary_opening(cleaned_mask, structure=np.ones((3, 3)))

    # Label connected regions
    labeled_mask, num_features = label(cleaned_mask)

    # Find bounding boxes
    regions = []
    for i in range(1, num_features + 1):
        rows, cols = np.where(labeled_mask == i)

        if len(rows) == 0:
            continue

        y1, y2 = np.min(rows), np.max(rows)
        x1, x2 = np.min(cols), np.max(cols)

        # Calculate aspect ratio
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height != 0 else 0

        # Filter by size and aspect ratio
        if (width * height > 100 and
                0.5 < aspect_ratio < 2.0 and
                width > 20 and height > 20):
            regions.append((x1, y1, x2, y2))

    return regions