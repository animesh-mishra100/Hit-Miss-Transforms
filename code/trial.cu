%%writefile image_processing.cu
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdexcept>

// Constants for kernel configurations
#define BLOCK_SIZE 16
#define KERNEL_RADIUS 3
#define HOURGLASS_SIZE 7

// Constant memory declarations
__constant__ int d_hourglass[4][HOURGLASS_SIZE][HOURGLASS_SIZE];
__constant__ float d_sharpen_kernel[3][3];

// Initialize constant memory data on host
void initializeConstants() {
    // Initialize hourglass patterns for membrane detection
    int h_hourglass[4][HOURGLASS_SIZE][HOURGLASS_SIZE] = {
        // Vertical hourglass
        {
            {0,0,0,1,0,0,0},
            {0,0,1,1,1,0,0},
            {0,1,1,1,1,1,0},
            {1,1,1,0,1,1,1},
            {0,1,1,1,1,1,0},
            {0,0,1,1,1,0,0},
            {0,0,0,1,0,0,0}
        },
        // Horizontal hourglass
        {
            {0,0,0,1,0,0,0},
            {0,0,1,1,1,0,0},
            {0,1,0,0,0,1,0},
            {1,1,0,0,0,1,1},
            {0,1,0,0,0,1,0},
            {0,0,1,1,1,0,0},
            {0,0,0,1,0,0,0}
        },
        // Diagonal hourglass (45 degrees)
        {
            {1,0,0,0,0,0,0},
            {0,1,0,0,0,0,0},
            {0,0,1,0,0,0,0},
            {0,0,0,0,0,0,0},
            {0,0,0,0,1,0,0},
            {0,0,0,0,0,1,0},
            {0,0,0,0,0,0,1}
        },
        // Diagonal hourglass (135 degrees)
        {
            {0,0,0,0,0,0,1},
            {0,0,0,0,0,1,0},
            {0,0,0,0,1,0,0},
            {0,0,0,0,0,0,0},
            {0,0,1,0,0,0,0},
            {0,1,0,0,0,0,0},
            {1,0,0,0,0,0,0}
        }
    };

    // Initialize sharpening kernel
    float h_sharpen_kernel[3][3] = {
        {-1, -1, -1},
        {-1,  9, -1},
        {-1, -1, -1}
    };

    // Copy to constant memory
    cudaMemcpyToSymbol(d_hourglass, h_hourglass, 4 * HOURGLASS_SIZE * HOURGLASS_SIZE * sizeof(int));
    cudaMemcpyToSymbol(d_sharpen_kernel, h_sharpen_kernel, 9 * sizeof(float));
}

// Device functions for helper operations
__device__ float computeLocalMean(int x, int y, int width, int height, 
                                cudaTextureObject_t texObj,
                                const int mask[HOURGLASS_SIZE][HOURGLASS_SIZE]) {
    float sum = 0.0f;
    int count = 0;
    
    for(int i = -HOURGLASS_SIZE/2; i <= HOURGLASS_SIZE/2; i++) {
        for(int j = -HOURGLASS_SIZE/2; j <= HOURGLASS_SIZE/2; j++) {
            if(x + i >= 0 && x + i < height && y + j >= 0 && y + j < width) {
                if(mask[i + HOURGLASS_SIZE/2][j + HOURGLASS_SIZE/2] > 0) {
                    sum += tex2D<unsigned char>(texObj, y + j, x + i);
                    count++;
                }
            }
        }
    }
    return (count > 0) ? sum / count : 0;
}

// Main CUDA kernel for cancer cell detection
__global__ void detectCancerCellsKernel(
    cudaTextureObject_t texObj,
    unsigned char* preprocessed,
    unsigned char* membranes,
    unsigned char* nuclei,
    unsigned char* cancerCells,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(x >= height || y >= width) return;
    
    int idx = x * width + y;
    
    // 1. Preprocess image
    float pixel = tex2D<unsigned char>(texObj, y, x);
    float sharpened = 0.0f;
    
    // Apply sharpening
    for(int i = -1; i <= 1; i++) {
        for(int j = -1; j <= 1; j++) {
            if(x + i >= 0 && x + i < height && y + j >= 0 && y + j < width) {
                sharpened += tex2D<unsigned char>(texObj, y + j, x + i) * 
                            d_sharpen_kernel[i+1][j+1];
            }
        }
    }
    preprocessed[idx] = min(max((int)sharpened, 0), 255);
    
    // 2. Detect membranes using hourglass elements
    float maxResponse = 0.0f;
    for(int h = 0; h < 4; h++) {
        float response = computeLocalMean(x, y, width, height, texObj, d_hourglass[h]);
        maxResponse = max(maxResponse, response);
    }
    membranes[idx] = (maxResponse > 25.5f) ? 255 : 0;
    
    // 3. Detect nuclei
    float blueChannel = tex2D<unsigned char>(texObj, y, x);
    nuclei[idx] = (blueChannel < 100) ? 255 : 0;
    
    // 4. Combine membrane and nuclei detection
    bool isMembrane = membranes[idx] > 0;
    bool isNucleus = nuclei[idx] > 0;
    
    // Local area analysis for cancer cell detection
    int neighborCount = 0;
    for(int i = -2; i <= 2; i++) {
        for(int j = -2; j <= 2; j++) {
            if(x + i >= 0 && x + i < height && y + j >= 0 && y + j < width) {
                if(membranes[(x + i) * width + (y + j)] > 0 && 
                   nuclei[(x + i) * width + (y + j)] > 0) {
                    neighborCount++;
                }
            }
        }
    }
    
    cancerCells[idx] = (isMembrane && isNucleus && neighborCount >= 5) ? 255 : 0;
}

// Host function to process image
void processCancerDetection(const cv::Mat& input, cv::Mat& output) {
    int width = input.cols;
    int height = input.rows;
    size_t size = width * height * sizeof(unsigned char);
    
    // Initialize constant memory
    initializeConstants();
    
    // Allocate device memory
    unsigned char *d_preprocessed, *d_membranes, *d_nuclei, *d_cancerCells;
    cudaError_t status;
    
    status = cudaMalloc(&d_preprocessed, size);
    if (status != cudaSuccess) {
        throw std::runtime_error("Failed to allocate preprocessed memory");
    }
    
    status = cudaMalloc(&d_membranes, size);
    if (status != cudaSuccess) {
        cudaFree(d_preprocessed);
        throw std::runtime_error("Failed to allocate membranes memory");
    }
    
    status = cudaMalloc(&d_nuclei, size);
    if (status != cudaSuccess) {
        cudaFree(d_preprocessed);
        cudaFree(d_membranes);
        throw std::runtime_error("Failed to allocate nuclei memory");
    }
    
    status = cudaMalloc(&d_cancerCells, size);
    if (status != cudaSuccess) {
        cudaFree(d_preprocessed);
        cudaFree(d_membranes);
        cudaFree(d_nuclei);
        throw std::runtime_error("Failed to allocate cancer cells memory");
    }
    
    // Create CUDA array and copy image
    cudaArray* cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
    status = cudaMallocArray(&cuArray, &channelDesc, width, height);
    if (status != cudaSuccess) {
        cudaFree(d_preprocessed);
        cudaFree(d_membranes);
        cudaFree(d_nuclei);
        cudaFree(d_cancerCells);
        throw std::runtime_error("Failed to allocate CUDA array");
    }
    
    status = cudaMemcpy2DToArray(cuArray, 0, 0, input.data, input.step,
                                width * sizeof(unsigned char), height,
                                cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFreeArray(cuArray);
        cudaFree(d_preprocessed);
        cudaFree(d_membranes);
        cudaFree(d_nuclei);
        cudaFree(d_cancerCells);
        throw std::runtime_error("Failed to copy data to CUDA array");
    }
    
    // Create texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    
    cudaTextureObject_t texObj = 0;
    status = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    if (status != cudaSuccess) {
        cudaFreeArray(cuArray);
        cudaFree(d_preprocessed);
        cudaFree(d_membranes);
        cudaFree(d_nuclei);
        cudaFree(d_cancerCells);
        throw std::runtime_error("Failed to create texture object");
    }
    
    // Configure kernel launch parameters
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((height + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (width + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    detectCancerCellsKernel<<<gridSize, blockSize>>>(
        texObj, d_preprocessed, d_membranes, d_nuclei, d_cancerCells,
        width, height
    );
    
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);
        cudaFree(d_preprocessed);
        cudaFree(d_membranes);
        cudaFree(d_nuclei);
        cudaFree(d_cancerCells);
        throw std::runtime_error("Kernel launch failed");
    }
    
    // Copy results back to host
    output = cv::Mat(height, width, CV_8UC1);
    status = cudaMemcpy(output.data, d_cancerCells, size, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        cudaDestroyTextureObject(texObj);
        cudaFreeArray(cuArray);
        cudaFree(d_preprocessed);
        cudaFree(d_membranes);
        cudaFree(d_nuclei);
        cudaFree(d_cancerCells);
        throw std::runtime_error("Failed to copy results back to host");
    }
    
    // Cleanup
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(cuArray);
    cudaFree(d_preprocessed);
    cudaFree(d_membranes);
    cudaFree(d_nuclei);
    cudaFree(d_cancerCells);
}
int main() {
    try {
        // Load input image with explicit path and error checking
        std::string image_path = "content/usage.png";
        cv::Mat input = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        
        if(input.empty()) {
            // Print current working directory to help with debugging
            char cwd[PATH_MAX];
            if (getcwd(cwd, sizeof(cwd)) != NULL) {
                std::cerr << "Current working directory: " << cwd << std::endl;
            }
            
            std::cerr << "Could not load image at path: " << image_path << std::endl;
            std::cerr << "Please ensure the image exists and the path is correct" << std::endl;
            return -1;
        }

        std::cout << "Successfully loaded image with dimensions: " 
                  << input.cols << "x" << input.rows << std::endl;
        
        // Process image on GPU
        cv::Mat output;
        processCancerDetection(input, output);
        
        // Save results with absolute path
        std::string output_path = "cancer_cells_detected.png";
        bool success = cv::imwrite(output_path, output);
        if (!success) {
            std::cerr << "Failed to save output image" << std::endl;
            return -1;
        }
        
        std::cout << "Results saved to: " << output_path << std::endl;
        
        // Display results (if running in GUI environment)
        cv::imshow("Cancer Cells Detected", output);
        cv::waitKey(0);
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
        return -1;
    } catch (const std::runtime_error& e) {
        std::cerr << "Runtime Error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}