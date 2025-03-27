import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.spatial.distance import cdist


class SpeedSignClassifier:
    def __init__(self, descriptor_file="1-NN-descriptor-vects.npy"):
        """
        Initialize the 1-NN classifier with exemplar vectors.
        """
        mat = np.load(descriptor_file)
        self.categories = mat[:, 0]
        self.template_vectors = mat[:, 1:]

    def preprocess_roi(self, roi):
        """
        Convert an RoI to a normalized descriptor vector.
        """
        # Convert to grayscale and resize
        gray = rgb2gray(roi)
        resized = resize(gray, (64, 64), anti_aliasing=True)

        # Convert to 0-255 range and subtract mean
        normalized = (resized * 255).astype(np.float32)
        normalized -= np.mean(normalized)

        # Flatten and normalize to unit vector
        flattened = normalized.flatten()
        norm = np.linalg.norm(flattened)
        if norm > 0:
            flattened /= norm

        return flattened

    def classify(self, roi):
        """
        Classify an RoI using 1-NN.
        Returns speed value (40,50,60,80,100,120) or -1 if not a sign.
        """
        # Preprocess the RoI
        query_vector = self.preprocess_roi(roi)

        if len(query_vector) != 4096:
            return -1

        # Calculate distances to all exemplars
        distances = cdist(query_vector.reshape(1, -1),
                          self.template_vectors,
                          metric='euclidean')

        # Find nearest neighbor
        nearest_idx = np.argmin(distances)
        return int(self.categories[nearest_idx])