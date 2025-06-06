# Import required libraries
import cv2  # OpenCV for image processing
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
from scipy.ndimage import label  # For connected component labeling
import time  # For timing measurements

def create_hourglass_elements():
    """Create structural elements for membrane detection in the shape of hourglasses.
    Returns a list of 4 hourglass elements in different orientations."""
    hourglass_elements = []
    
    # Horizontal hourglass - detects horizontal membrane patterns
    h1 = np.array([
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [0,0,2,2,2,0,0],  # Center row contains foreground pixels (2)
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1]
    ])
    hourglass_elements.append(h1)
    
    # Vertical hourglass - detects vertical membrane patterns
    h2 = h1.T  # Transpose for vertical orientation
    hourglass_elements.append(h2)
    
    # Diagonal hourglasses - detect diagonal membrane patterns
    h3 = np.rot90(h1, k=1)  # 90 degree rotation
    h4 = np.rot90(h1, k=3)  # 270 degree rotation
    hourglass_elements.extend([h3, h4])
    
    return hourglass_elements

def preprocess_image(image):
    """Enhance image quality through contrast enhancement and sharpening.
    Args:
        image: Input grayscale image
    Returns:
        Preprocessed image with enhanced contrast and sharpness"""
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)

    # Apply sharpening using a kernel that enhances edges
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel) 

    return sharpened

def rank_grey_level_hit_or_miss(image, se, k_bg=0.7, h=0.1):
    """Apply rank-based grey level hit-or-miss transform.
    Args:
        image: Input image
        se: Structuring element
        k_bg: Background threshold parameter
        h: Minimum difference threshold
    Returns:
        Binary image highlighting matched patterns"""
    fg = se == 2  # Foreground elements
    bg = se == 1  # Background elements
    
    # Calculate mean of foreground pixels
    fg_mean = cv2.filter2D(image.astype(float), -1, fg.astype(float) / np.sum(fg))
    bg_values = cv2.filter2D(image.astype(float), -1, bg.astype(float))
    bg_rank = np.percentile(bg_values, k_bg * 100, axis=(0,1))
    
    # Apply thresholds to identify matching patterns
    result = np.logical_and(fg_mean > bg_rank, fg_mean - bg_rank > h * 255)
    return result.astype(np.uint8) * 255

def detect_membranes(image):
    """Detect cell membranes using edge detection and morphological operations.
    Args:
        image: Input preprocessed image
    Returns:
        Binary image with detected membrane edges"""
    # Apply Gaussian blur to reduce noise while preserving edges
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Remove background using morphological opening
    kernel = np.ones((9, 9), np.uint8)  # Adjust kernel size as needed
    background = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    foreground = cv2.subtract(blurred, background)

    # Create binary mask by thresholding
    _, thresh = cv2.threshold(foreground, 30, 255, cv2.THRESH_BINARY)

    # Detect edges using Canny algorithm
    edges = cv2.Canny(thresh, 50, 150)  # Adjust thresholds as needed

    # Connect broken edges using morphological closing
    kernel = np.ones((5, 5), np.uint8) 
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return closed

def detect_nuclei(image):
    """Detect cell nuclei using thresholding and morphological operations.
    Args:
        image: Input image (color or grayscale)
    Returns:
        Binary mask of detected nuclei"""
    # Handle both color and grayscale images
    if len(image.shape) == 3:  # Color image
        blue_channel = image[:,:,0]  # Use blue channel for nuclei detection
        _, nuclei_mask = cv2.threshold(blue_channel, 100, 255, cv2.THRESH_BINARY_INV)
    else:  # Grayscale image
        _, nuclei_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up nuclei mask using morphological operations
    nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    nuclei_mask = cv2.dilate(nuclei_mask, np.ones((3,3), np.uint8), iterations=1)

    return nuclei_mask

def filter_cancer_cells(membranes, nuclei, min_neighbors=3):
    """Identify potential cancer cells by combining membrane and nuclei detection results.
    Args:
        membranes: Binary image of detected membranes
        nuclei: Binary image of detected nuclei
        min_neighbors: Minimum size threshold for cancer cell regions
    Returns:
        Binary image highlighting potential cancer cells"""
    # Find regions where both membranes and nuclei are present
    cancer_cells = cv2.bitwise_and(membranes, nuclei) 
    
    # Label connected components
    labeled_array, num_features = label(cancer_cells)
    
    # Filter small regions that likely aren't cancer cells
    filtered_cancer_cells = np.zeros_like(cancer_cells)
    for region_label in range(1, num_features + 1):
        region = labeled_array == region_label
        neighbor_count = np.sum(region)
        if neighbor_count >= min_neighbors:
            filtered_cancer_cells[region] = 255

    return filtered_cancer_cells

def visualize_results(original, filtered_cancer_cells):
    """Display original image and detected cancer cells side by side.
    Args:
        original: Original input image
        filtered_cancer_cells: Final cancer cell detection result"""
    plt.figure(figsize=(10, 4))
    plt.subplot(121), plt.imshow(original, cmap='gray'), plt.title('Original')
    
    # Create color overlay of detected cancer cells
    result = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    result[filtered_cancer_cells > 0] = [0, 255, 0]  # Green for cancer cells
    
    plt.subplot(122), plt.imshow(result), plt.title('Detected Cancer Cells')
    
    plt.tight_layout()
    plt.show()

# Main execution block
image_path = '/Users/anikethhebbar/Desktop/workspace/MTech/HPCS/OpenMP/Screenshot 2024-10-16 at 22.27.03.png'
#image_path = "test2.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale

# Time and perform preprocessing
start_time = time.time()
preprocessed = preprocess_image(image)
preprocess_time = time.time() - start_time
print(f"Preprocessing time: {preprocess_time:.4f} seconds")

# Time and perform membrane detection
start_time = time.time()
membranes = detect_membranes(preprocessed)
membrane_detection_time = time.time() - start_time
print(f"Membrane detection time: {membrane_detection_time:.4f} seconds")

# Time and perform nuclei detection
start_time = time.time()
nuclei = detect_nuclei(preprocessed)
nuclei_detection_time = time.time() - start_time
print(f"Nuclei detection time: {nuclei_detection_time:.4f} seconds")

# Time and perform cancer cell filtering
start_time = time.time()
filtered_cancer_cells = filter_cancer_cells(membranes, nuclei, min_neighbors=200)
filtering_time = time.time() - start_time
print(f"Filtering cancer cells time: {filtering_time:.4f} seconds")

# Calculate and display total processing time
total_time = preprocess_time + membrane_detection_time + nuclei_detection_time + filtering_time
print(f"Total processing time: {total_time:.4f} seconds")

# Display final results
visualize_results(image, filtered_cancer_cells)