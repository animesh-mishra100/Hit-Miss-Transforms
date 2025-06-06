# Import required libraries
from mpi4py import MPI  # For parallel processing
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
    kernel = np.ones((9, 9), np.uint8)  # Large kernel for background estimation
    background = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    foreground = cv2.subtract(blurred, background)

    # Create binary mask of potential membrane regions
    _, thresh = cv2.threshold(foreground, 30, 255, cv2.THRESH_BINARY)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(thresh, 50, 150)

    # Connect broken edges using morphological closing
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return closed

def detect_nuclei(image):
    """Detect cell nuclei using thresholding and morphological operations.
    Args:
        image: Input preprocessed image
    Returns:
        Binary mask of detected nuclei"""
    # Apply appropriate thresholding based on image type
    if len(image.shape) == 3:  # For color images
        blue_channel = image[:,:,0]  # Extract blue channel (BGR format)
        _, nuclei_mask = cv2.threshold(blue_channel, 100, 255, cv2.THRESH_BINARY_INV)
    else:  # For grayscale images
        _, nuclei_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up nuclei mask using morphological operations
    nuclei_mask = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    nuclei_mask = cv2.dilate(nuclei_mask, np.ones((3,3), np.uint8), iterations=1)

    return nuclei_mask

def filter_cancer_cells(membranes, nuclei, min_neighbors=3):
    """Filter detected cancer cells based on neighborhood analysis.
    Args:
        membranes: Binary image of detected membranes
        nuclei: Binary image of detected nuclei
        min_neighbors: Minimum number of neighboring pixels to keep a region
    Returns:
        Binary image of filtered cancer cells"""
    # Combine membrane and nuclei detections
    cancer_cells = cv2.bitwise_and(membranes, nuclei)
    
    # Label connected regions for analysis
    labeled_array, num_features = label(cancer_cells)
    
    # Filter regions based on neighborhood size
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
        filtered_cancer_cells: Binary mask of detected cancer cells"""
    plt.figure(figsize=(20, 4))
    plt.subplot(121), plt.imshow(original, cmap='gray'), plt.title('Original Image')
    
    # Create color overlay of detected cancer cells
    result = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    result[filtered_cancer_cells > 0] = [0, 255, 0]  # Green for detected cancer cells
    
    plt.subplot(122), plt.imshow(result), plt.title('Detected Cancer Cells')
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function implementing parallel image processing using MPI."""
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Load image (master process only)
    image_path = '/Users/anikethhebbar/Desktop/workspace/MTech/HPCS/OpenMP/Screenshot 2024-10-16 at 22.27.03.png'
    if rank == 0:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image at {image_path}")
            comm.Abort()
        height, width = image.shape
    else:
        image = None
        height = width = 0

    # Broadcast image dimensions to all processes
    height = comm.bcast(height if rank == 0 else None, root=0)
    width = comm.bcast(width if rank == 0 else None, root=0)

    # Divide image into slices for parallel processing
    slices = np.array_split(np.arange(height), size)
    local_indices = slices[rank]
    local_height = len(local_indices)

    # Distribute image slices to all processes
    if rank == 0:
        sendbuf = [image[local_indices, :] for local_indices in slices]
    else:
        sendbuf = None

    # Time the scatter operation
    start_time = time.time()
    local_image = comm.scatter(sendbuf, root=0)
    scatter_time = time.time() - start_time

    # Process local image slice
    start_time = time.time()
    preprocessed = preprocess_image(local_image)
    membranes = detect_membranes(preprocessed)
    nuclei = detect_nuclei(preprocessed)
    filtered_cancer_cells = filter_cancer_cells(membranes, nuclei, min_neighbors=100)
    processing_time = time.time() - start_time

    # Gather results back to master process
    start_time = time.time()
    gathered_preprocessed = comm.gather(preprocessed, root=0)
    gathered_membranes = comm.gather(membranes, root=0)
    gathered_nuclei = comm.gather(nuclei, root=0)
    gathered_filtered = comm.gather(filtered_cancer_cells, root=0)
    gather_time = time.time() - start_time

    # Master process combines results and displays output
    if rank == 0:
        # Reconstruct full images from gathered slices
        full_preprocessed = np.vstack(gathered_preprocessed)
        full_membranes = np.vstack(gathered_membranes)
        full_nuclei = np.vstack(gathered_nuclei)
        full_filtered = np.vstack(gathered_filtered)

        # Display results and timing information
        visualize_results(image, full_filtered)
        print(f"Scatter time: {scatter_time:.4f} seconds")
        print(f"Processing time: {processing_time:.4f} seconds")
        print(f"Gather time: {gather_time:.4f} seconds")
        total_time = scatter_time + processing_time + gather_time
        print(f"Total time: {total_time:.4f} seconds")

if __name__ == "__main__":
    main()
