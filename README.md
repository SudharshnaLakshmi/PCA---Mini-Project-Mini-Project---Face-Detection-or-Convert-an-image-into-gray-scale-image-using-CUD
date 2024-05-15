# PCA-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD
Mini Project - Face Detection or Convert an image into gray scale image using CUDA GPU programming
## AIM:
The aim of this project is to convert a color image to grayscale using parallel computing with CUDA in a Google Colab environment. 

## PROCEDURE:
1. Load the input color image from a file (input.jpg).
2. Check if the image is successfully loaded. If not, print an error message and exit.
3. Extract the dimensions (rows and columns) of the image.
4. Allocate memory for the RGB image and copy the image data into it.
5. Calculate the sizes of the input RGB image and the output grayscale image.
6. Allocate memory on the device (GPU) for both the RGB and grayscale images using cudaMalloc.
7. Copy the RGB image data from the host (CPU) to the device (GPU) memory using cudaMemcpy.
8. Define the CUDA kernel colorConvertToGray to convert RGB pixels to grayscale.
9. Configure the grid and block dimensions for CUDA kernel invocation.
10. Launch the CUDA kernel colorConvertToGray with the configured grid and block dimensions.
11. Copy the resulting grayscale image data from the device memory to the host memory using cudaMemcpy.
12. Save the grayscale image to a file (output.jpg) using OpenCV.
13. Free the allocated memory on the device and the host using cudaFree and free, respectively.

## PROGRAM:
```
!apt-get update
!apt-get install -y nvidia-cuda-toolkit
!pip install opencv-python

%%writefile grayscale.cu

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define CHANNELS 3

__global__ 
void colorConvertToGray(unsigned char *rgb, unsigned char *gray, int rows, int cols) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (col < cols && row < rows) {
        int gray_offset = row * cols + col;
        int rgb_offset = gray_offset * CHANNELS;

        unsigned char r = rgb[rgb_offset];
        unsigned char g = rgb[rgb_offset + 1];
        unsigned char b = rgb[rgb_offset + 2];

        gray[gray_offset] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

void loadImageFile(unsigned char **rgb_image, int *rows, int *cols, const std::string &file) {
    cv::Mat img = cv::imread(file, cv::IMREAD_COLOR);
    if (img.empty()) {
        fprintf(stderr, "Error: Unable to load image %s\n", file.c_str());
        exit(EXIT_FAILURE);
    }

    *rows = img.rows;
    *cols = img.cols;

    *rgb_image = (unsigned char*) malloc(*rows * *cols * CHANNELS * sizeof(unsigned char));
    memcpy(*rgb_image, img.data, *rows * *cols * CHANNELS * sizeof(unsigned char));
}

void saveImageFile(const unsigned char *gray_image, int rows, int cols, const std::string &file) {
    cv::Mat img(rows, cols, CV_8UC1, (void*)gray_image);
    cv::imwrite(file, img);
}

int main() {
    std::string input_file = "input.jpg";
    std::string output_file = "output.jpg";

    unsigned char *h_rgb_image, *h_gray_image;
    unsigned char *d_rgb_image, *d_gray_image;
    int rows, cols;

    loadImageFile(&h_rgb_image, &rows, &cols, input_file);

    size_t image_size = rows * cols * CHANNELS * sizeof(unsigned char);
    size_t gray_image_size = rows * cols * sizeof(unsigned char);

    h_gray_image = (unsigned char*) malloc(gray_image_size);

    cudaMalloc(&d_rgb_image, image_size);
    cudaMalloc(&d_gray_image, gray_image_size);

    cudaMemcpy(d_rgb_image, h_rgb_image, image_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);
    
    colorConvertToGray<<<dimGrid, dimBlock>>>(d_rgb_image, d_gray_image, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_gray_image, d_gray_image, gray_image_size, cudaMemcpyDeviceToHost);

    saveImageFile(h_gray_image, rows, cols, output_file);

    cudaFree(d_rgb_image);
    cudaFree(d_gray_image);
    free(h_rgb_image);
    free(h_gray_image);

    return 0;
}

!nvcc -o grayscale grayscale.cu `pkg-config --cflags --libs opencv4`
!./grayscale

import cv2
from matplotlib import pyplot as plt
output_image = cv2.imread('output.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(output_image, cmap='gray')
plt.axis('off')
plt.show()
```
## OUTPUT:
### COLOR IMAGE :
![](input.jpg)

### GRAY SCALE IMAGE:
![](output.png)

## RESULT:
The color image (input.jpg) is converted to grayscale.The resulting grayscale image is saved as output.jpg.Display the resulting grayscale image using matplotlib for visualization.
