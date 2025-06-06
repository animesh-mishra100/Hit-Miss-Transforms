# Import required libraries
from mpi4py import MPI  # For parallel processing using Message Passing Interface
import cv2  # OpenCV for image processing and computer vision tasks
import numpy as np  # For numerical operations and array manipulation
import matplotlib.pyplot as plt  # For visualization and plotting results
from scipy.ndimage import label  # For connected component labeling in image analysis


def create_hourglass_elements():
    """Create structural elements for membrane detection in the shape of hourglasses.
    These elements are used as templates to detect membrane patterns in different orientations.
    Returns a list of 4 hourglass elements in different orientations (horizontal, vertical, and two diagonals)."""
    hourglass_elements = []

    # Horizontal hourglass - detects horizontal membrane patterns
    # The 2's indicate foreground pixels and 1's indicate background pixels
    # The shape resembles an hourglass lying on its side
    h1 = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 2, 1, 0, 0],  # Center row with foreground pixel (2)
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0]
        ]
    )
    hourglass_elements.append(h1)

    # Vertical hourglass - detects vertical membrane patterns
    # Created by transposing the horizontal hourglass
    h2 = h1.T  # Transpose for vertical orientation
    hourglass_elements.append(h2)

    # Diagonal hourglasses - detect diagonal membrane patterns
    # Created by rotating the horizontal hourglass
    h3 = np.rot90(h1, k=1)  # 90 degree rotation for diagonal orientation
    h4 = np.rot90(h1, k=3)  # 270 degree rotation for opposite diagonal
    hourglass_elements.extend([h3, h4])

    return hourglass_elements


def preprocess_image(image):
    """Enhance image quality through contrast enhancement and sharpening.
    This improves the visibility of cell membranes and other features.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Preprocessed image with enhanced contrast and sharpness"""
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement
    # This improves local contrast while preventing noise amplification
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Apply sharpening using a kernel that enhances edges
    # The kernel has a high center value (9) surrounded by -1s
    # This emphasizes differences between neighboring pixels
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


def rank_grey_level_hit_or_miss(image, se, k_bg=0.5, h=0.1):
    """Apply rank-based grey level hit-or-miss transform.
    This transform identifies structures that match the given structural element pattern.
    
    Args:
        image: Input image
        se: Structuring element (template pattern to match)
        k_bg: Background threshold parameter (0-1)
        h: Minimum difference threshold
        
    Returns:
        Binary image highlighting matched patterns"""
    # Separate foreground (2) and background (1) elements from structural element
    fg = se == 2  # Foreground elements (pattern to match)
    bg = se == 1  # Background elements (surrounding context)

    # Calculate mean of foreground pixels using convolution
    fg_mean = cv2.filter2D(image.astype(float), -1, fg.astype(float) / np.sum(fg))
    # Calculate background values for ranking
    bg_values = cv2.filter2D(image.astype(float), -1, bg.astype(float))
    # Get background threshold at specified percentile
    bg_rank = np.percentile(bg_values, k_bg * 100, axis=(0, 1))

    # Apply thresholds to identify matching patterns:
    # 1. Foreground mean must be greater than background rank
    # 2. Difference must exceed minimum threshold
    result = np.logical_and(fg_mean > bg_rank, fg_mean - bg_rank > h * 255)
    return result.astype(np.uint8) * 255


def detect_membranes(image):
    """Detect cell membranes using edge detection, morphological operations, and rank-based hit-or-miss.
    This combines multiple techniques to robustly identify cell boundaries.
    
    Args:
        image: Input preprocessed image
        
    Returns:
        Binary image with detected membrane edges"""
    # Apply Gaussian blur to reduce noise while preserving edges
    # 5x5 kernel provides moderate smoothing
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Remove background using morphological opening
    # Large 9x9 kernel helps estimate broad background variations
    kernel = np.ones((9, 9), np.uint8)  # Large kernel for background estimation
    background = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    # Subtract background to enhance foreground features
    foreground = cv2.subtract(blurred, background)

    # Create binary mask of potential membrane regions
    # Threshold of 30 separates membrane features from background
    _, thresh = cv2.threshold(foreground, 30, 255, cv2.THRESH_BINARY)

    # Detect edges in membrane regions using Canny edge detection
    # Moderate thresholds (50,150) balance sensitivity and noise
    edges = cv2.Canny(thresh, 50, 150)

    # Connect nearby edges using morphological closing
    # 5x5 kernel bridges small gaps in membrane boundaries
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Apply rank-based grey level hit-or-miss transform with hourglass elements
    # This detects membrane patterns in multiple orientations
    hourglass_elements = create_hourglass_elements()
    hit_or_miss_result = np.zeros_like(closed)
    for se in hourglass_elements:
        matched = rank_grey_level_hit_or_miss(image, se, k_bg=0.5, h=0.1)
        hit_or_miss_result = cv2.bitwise_or(hit_or_miss_result, matched)

    # Combine Canny edges with hit-or-miss results for final membrane detection
    combined_membranes = cv2.bitwise_or(closed, hit_or_miss_result)

    return combined_membranes


def detect_nuclei(image):
    """Detect cell nuclei using thresholding and morphological operations.
    Nuclei typically appear as darker regions in brightfield microscopy.
    
    Args:
        image: Input image (color or grayscale)
        
    Returns:
        Binary mask of detected nuclei"""
    # Handle both color and grayscale images differently
    if len(image.shape) == 3:  # Color image
        # Blue channel often provides best contrast for nuclei
        blue_channel = image[:, :, 0]  # Use blue channel for nuclei detection
        _, nuclei_mask = cv2.threshold(blue_channel, 100, 255, cv2.THRESH_BINARY_INV)
    else:  # Grayscale image
        # Use Otsu's method for automatic threshold selection
        _, nuclei_mask = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    # Clean up nuclei mask using morphological operations
    # Opening removes small noise
    nuclei_mask = cv2.morphologyEx(
        nuclei_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    # Dilation slightly enlarges nuclei regions
    nuclei_mask = cv2.dilate(nuclei_mask, np.ones((3, 3), np.uint8), iterations=1)

    return nuclei_mask


def filter_cancer_cells(membranes, nuclei, min_neighbors=50):
    """Identify potential cancer cells by combining membrane and nuclei detection results.
    Cancer cells often show distinct membrane and nuclear patterns.
    
    Args:
        membranes: Binary image of detected membranes
        nuclei: Binary image of detected nuclei
        min_neighbors: Minimum size threshold for cancer cell regions
        
    Returns:
        Binary image highlighting potential cancer cells"""
    # Find regions where both membranes and nuclei are present
    # These overlapping regions are candidates for cancer cells
    cancer_cells = cv2.bitwise_and(membranes, nuclei)

    # Label connected components to identify distinct regions
    labeled_array, num_features = label(cancer_cells)

    # Filter small regions that likely aren't cancer cells
    # Only keep regions larger than min_neighbors threshold
    filtered_cancer_cells = np.zeros_like(cancer_cells)
    for region_label in range(1, num_features + 1):
        region = labeled_array == region_label
        neighbor_count = np.sum(region)
        if neighbor_count >= min_neighbors:
            filtered_cancer_cells[region] = 255

    return filtered_cancer_cells


def visualize_results(original, preprocessed, membranes, nuclei, filtered_cancer_cells):
    """Display original image and detected cancer cells side by side.
    Creates a visualization comparing input and results.
    
    Args:
        original: Original input image
        preprocessed, membranes, nuclei: Intermediate processing results
        filtered_cancer_cells: Final cancer cell detection result"""
    plt.figure(figsize=(12, 6))

    # Display original image on the left
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(f"Original Image ({original.shape[0]}x{original.shape[1]})")

    # Create result image with cancer cells highlighted in green
    result = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    result[filtered_cancer_cells > 0] = [0, 255, 0]  # Green highlighting

    # Display result image on the right
    plt.subplot(122)
    plt.imshow(result)
    plt.title("Detected Cancer Cells")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    """Main function implementing parallel image processing using MPI.
    Distributes image processing across multiple processes for faster computation."""
    # Initialize MPI environment for parallel processing
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Process ID (0 is master, others are workers)
    size = comm.Get_size()  # Total number of processes

    # Image loading (master process only)
    #image_path = "/Users/anikethhebbar/Desktop/workspace/MTech/HPCS/OpenMP/Screenshot 2024-10-16 at 22.27.03.png"
    image_path = "image1.png"
    if rank == 0:
        # Master process loads the image
        image = cv2.imread(image_path)  # Read in color mode
        if image is None:
            print(f"Failed to load image at {image_path}")
            comm.Abort()
        height, width = image.shape[:2]

        # Convert to grayscale for processing while keeping original in color
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Divide image into horizontal slices for parallel processing
        # Each process gets approximately equal sized slice
        slices = np.array_split(np.arange(height), size)

        # Distribute slices to worker processes
        for i in range(1, size):
            slice_indices = slices[i]
            slice_data = image_gray[slice_indices, :]
            comm.send((slice_indices, slice_data), dest=i, tag=11)

        # Master process takes first slice
        local_indices = slices[0]
        local_image = image_gray[local_indices, :]

    else:
        # Worker processes receive their slice from master
        local_indices, local_image = comm.recv(source=0, tag=11)
        height, width = None, None
        image = None

    # Process local slice - each process handles its portion
    start_time = MPI.Wtime()
    preprocessed = preprocess_image(local_image)
    membranes = detect_membranes(preprocessed)
    nuclei = detect_nuclei(preprocessed)
    filtered_cancer_cells = filter_cancer_cells(membranes, nuclei, min_neighbors=210)
    processing_time = MPI.Wtime() - start_time

    # Gather results from all processes
    if rank == 0:
        # Master process collects results
        gathered_preprocessed = [preprocessed]
        gathered_membranes = [membranes]
        gathered_nuclei = [nuclei]
        gathered_filtered = [filtered_cancer_cells]

        # Receive results from all worker processes
        for i in range(1, size):
            recv_data = comm.recv(source=i, tag=22)
            gathered_preprocessed.append(recv_data["preprocessed"])
            gathered_membranes.append(recv_data["membranes"])
            gathered_nuclei.append(recv_data["nuclei"])
            gathered_filtered.append(recv_data["filtered_cancer_cells"])

        # Reconstruct full images by combining slices
        full_preprocessed = np.vstack(gathered_preprocessed)
        full_membranes = np.vstack(gathered_membranes)
        full_nuclei = np.vstack(gathered_nuclei)
        full_filtered = np.vstack(gathered_filtered)

        # Display final results and processing time
        visualize_results(
            image, full_preprocessed, full_membranes, full_nuclei, full_filtered
        )
        print(f"Processing time: {processing_time:.4f} seconds")
    else:
        # Workers send their results back to master
        comm.send(
            {
                "preprocessed": preprocessed,
                "membranes": membranes,
                "nuclei": nuclei,
                "filtered_cancer_cells": filtered_cancer_cells,
            },
            dest=0,
            tag=22,
        )


if __name__ == "__main__":
    main()
