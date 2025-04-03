import numpy as np
from skimage.color import rgb2hsv
from scipy.ndimage import label, binary_closing, binary_opening, binary_erosion
from skimage.transform import resize
import matplotlib.pyplot as plt

def propose_regions(rgb_image, huemin=0.949, huemax=0.048, sat_thresh=0.265, val_thresh=0.11, show_plots=False):
    hsv_image = rgb2hsv(rgb_image)

    # Use parameters for thresholds
    hue = hsv_image[..., 0]
    sat = hsv_image[..., 1]
    val = hsv_image[..., 2]

    hue_mask = (hue > huemin) | (hue < huemax)
    sat_mask = sat > sat_thresh
    val_mask = val > val_thresh

    combined_mask = hue_mask & sat_mask & val_mask

    # Morphological operations
    cleaned_mask = binary_closing(combined_mask, structure=np.ones((5, 5)))
    cleaned_mask = binary_opening(cleaned_mask, structure=np.ones((3, 3)))

    # Add binary erosion mask which culls pixels at the detection boundaries, preventing false positives
    eroded_mask = binary_erosion(cleaned_mask, structure=np.ones((3, 3)))
    labeled_mask, num_features = label(eroded_mask)
    
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
            
    # Add fallback: if no regions detected, use a dilated version of the original binary mask
    # so that pixels that are almost within the threshold and are adjacent to valid pixels are included.
    if not regions:
        from scipy.ndimage import binary_dilation
        fallback_mask = binary_dilation(combined_mask, structure=np.ones((3, 3)))
        fallback_label, fallback_num = label(fallback_mask)
        max_area = 0
        fallback_region = None
        for j in range(1, fallback_num + 1):
            rows, cols = np.where(fallback_label == j)
            if len(rows) == 0:
                continue
            x1, x2 = np.min(cols), np.max(cols)
            y1, y2 = np.min(rows), np.max(rows)
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            area = width * height
            if area > max_area and area > 100 and 0.5 < aspect_ratio < 2.0 and width > 20 and height > 20:
                max_area = area
                fallback_region = (x1, y1, x2, y2)
        # If any row or column in the fallback region is filled with 1s, diregard this region, since this is a straight line
        if fallback_region is not None:
            fx1, fy1, fx2, fy2 = fallback_region
            region_mask = fallback_mask[fy1:fy2, fx1:fx2]
            if np.any(np.sum(region_mask, axis=1) == region_mask.shape[1]) or \
               np.any(np.sum(region_mask, axis=0) == region_mask.shape[0]):
                fallback_region = None
        if fallback_region is not None:
            regions.append(fallback_region)

    if show_plots:
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
        plt.title("Eroded Mask")
        plt.imshow(eroded_mask, cmap='gray')
        plt.axis('off')
    
        plt.tight_layout()
        plt.show()
        
    return regions
