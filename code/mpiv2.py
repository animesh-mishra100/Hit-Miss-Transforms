# Import required libraries
from mpi4py import MPI  # For parallel processing
import cv2  # OpenCV for image processing
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualization
from scipy.ndimage import label  # For connected component labeling


def create_hourglass_elements():
    """Create structural elements for membrane detection in the shape of hourglasses.
    Returns a list of 4 hourglass elements in different orientations."""
    hourglass_elements = []

    # Horizontal hourglass - detects horizontal membrane patterns
    h1 = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 2, 1, 0, 0],  
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0]
        ]
    )
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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Apply sharpening using a kernel that enhances edges
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    return sharpened


def rank_grey_level_hit_or_miss(image, se, k_bg=0.5, h=0.1):
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
    bg_rank = np.percentile(bg_values, k_bg * 100, axis=(0, 1))

    # Apply thresholds to identify matching patterns
    result = np.logical_and(fg_mean > bg_rank, fg_mean - bg_rank > h * 255)
    return result.astype(np.uint8) * 255


def detect_membranes(image):
    """Detect cell membranes using edge detection, morphological operations, and rank-based hit-or-miss.
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

    # Detect edges in membrane regions
    edges = cv2.Canny(thresh, 50, 150)  # Canny edge detection with moderate thresholds

    # Connect nearby edges using morphological closing
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Apply rank-based grey level hit-or-miss transform with hourglass elements
    hourglass_elements = create_hourglass_elements()
    hit_or_miss_result = np.zeros_like(closed)
    for se in hourglass_elements:
        matched = rank_grey_level_hit_or_miss(image, se, k_bg=0.5, h=0.1)
        hit_or_miss_result = cv2.bitwise_or(hit_or_miss_result, matched)

    # Combine Canny edges with hit-or-miss results
    combined_membranes = cv2.bitwise_or(closed, hit_or_miss_result)

    return combined_membranes


def detect_nuclei(image):
    """Detect cell nuclei using thresholding and morphological operations.
    Args:
        image: Input image (color or grayscale)
    Returns:
        Binary mask of detected nuclei"""
    # Handle both color and grayscale images
    if len(image.shape) == 3:  # Color image
        blue_channel = image[:, :, 0]  # Use blue channel for nuclei detection
        _, nuclei_mask = cv2.threshold(blue_channel, 100, 255, cv2.THRESH_BINARY_INV)
    else:  # Grayscale image
        _, nuclei_mask = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    # Clean up nuclei mask using morphological operations
    nuclei_mask = cv2.morphologyEx(
        nuclei_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    nuclei_mask = cv2.dilate(nuclei_mask, np.ones((3, 3), np.uint8), iterations=1)

    return nuclei_mask


def filter_cancer_cells(membranes, nuclei, min_neighbors=50):
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


def visualize_results(original, preprocessed, membranes, nuclei, filtered_cancer_cells):
    """Display original image and detected cancer cells side by side.
    Args:
        original: Original input image
        preprocessed, membranes, nuclei: Intermediate processing results
        filtered_cancer_cells: Final cancer cell detection result"""
    plt.figure(figsize=(8, 8))

    # Original image
    plt.subplot(121), plt.imshow(original, cmap="gray"), plt.title("Original Image")

    # Overlay cancer cell detections in green
    result = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    result[filtered_cancer_cells > 0] = [0, 255, 0]  # Green highlighting

    plt.subplot(122), plt.imshow(result), plt.title("Detected Cancer Cells")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    """Main function implementing parallel image processing using MPI."""
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Process ID
    size = comm.Get_size()  # Total number of processes

    # Image loading (master process only)
    #image_path = "/Users/anikethhebbar/Desktop/workspace/MTech/HPCS/OpenMP/Screenshot 2024-10-16 at 22.27.03.png"
    image_path = "image1.png"
    if rank == 0:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image at {image_path}")
            comm.Abort()
        height, width = image.shape

        # Divide image into horizontal slices for parallel processing
        slices = np.array_split(np.arange(height), size)

        # Distribute slices to worker processes
        for i in range(1, size):
            slice_indices = slices[i]
            slice_data = image[slice_indices, :]
            comm.send((slice_indices, slice_data), dest=i, tag=11)

        # Master process takes first slice
        local_indices = slices[0]
        local_image = image[local_indices, :]

    else:
        # Worker processes receive their slice
        local_indices, local_image = comm.recv(source=0, tag=11)
        height, width = None, None

    # Process local slice
    start_time = MPI.Wtime()
    preprocessed = preprocess_image(local_image)
    membranes = detect_membranes(preprocessed)
    nuclei = detect_nuclei(preprocessed)
    filtered_cancer_cells = filter_cancer_cells(membranes, nuclei, min_neighbors=325)
    processing_time = MPI.Wtime() - start_time

    # Gather results
    if rank == 0:
        # Master process collects results
        gathered_preprocessed = [preprocessed]
        gathered_membranes = [membranes]
        gathered_nuclei = [nuclei]
        gathered_filtered = [filtered_cancer_cells]

        # Receive results from workers
        for i in range(1, size):
            recv_data = comm.recv(source=i, tag=22)
            gathered_preprocessed.append(recv_data["preprocessed"])
            gathered_membranes.append(recv_data["membranes"])
            gathered_nuclei.append(recv_data["nuclei"])
            gathered_filtered.append(recv_data["filtered_cancer_cells"])

        # Reconstruct full images
        full_preprocessed = np.vstack(gathered_preprocessed)
        full_membranes = np.vstack(gathered_membranes)
        full_nuclei = np.vstack(gathered_nuclei)
        full_filtered = np.vstack(gathered_filtered)

        # Display results
        visualize_results(
            image, full_preprocessed, full_membranes, full_nuclei, full_filtered
        )
        print(f"Processing time: {processing_time:.4f} seconds")
    else:
        # Workers send results back to master
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
