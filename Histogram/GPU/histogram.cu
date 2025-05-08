#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include "../../utils/dirent.h"
#include <string.h>
#include "../../utils/stb_image.h"
#include "../../utils/stb_image_write.h"
#include "histogram.h"
#include <cuda_runtime.h>
#include <time.h>

typedef struct {
    unsigned char *data;
    int width;
    int height;
    int channels;
    char filename[1024];
} Image;


// Combined RGB Histogram

// Kernel 1 - Atomic Add
__global__ void compute_histogram_atomic_add(unsigned char *data, unsigned int *histogram, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        atomicAdd(&histogram[data[tid]], 1);
    }
}

// Kernel 2 - Privatization
__global__ void compute_histogram_priv(unsigned char *data, unsigned int *histogram, int size) {
    __shared__ unsigned int private_histogram[256];

    // Initalized private versions to zero
    if (threadIdx.x < 256) {
        private_histogram[threadIdx.x] = 0;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (tid < size) {
        atomicAdd(&private_histogram[data[tid]], 1);
        tid += stride;
    }
    __syncthreads();

    if (threadIdx.x < 256) {
        atomicAdd(&histogram[threadIdx.x], private_histogram[threadIdx.x]);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <input_directory> <output_directory>\n", argv[0]);
        return 1;
    }

    DIR *dir = opendir(argv[1]);
    if (!dir) {
        perror("Cannot open directory\n");
        return 0;
    }

    struct dirent *entry;
    char filepath[1024];
    Image images[100]; 
    int images_count = 0;

    while ((entry = readdir(dir)) != NULL && images_count < 100) {
        if (entry->d_type == DT_REG) {
            snprintf(filepath, sizeof(filepath), "%s/%s", argv[1], entry->d_name);

            int width, height, channels;
            unsigned char *img_data = stbi_load(filepath, &width, &height, &channels, 0);

            if (img_data) {
                images[images_count].data = img_data;
                images[images_count].width = width;
                images[images_count].height = height;
                images[images_count].channels = channels;
                strncpy(images[images_count].filename, entry->d_name, sizeof(images[images_count].filename));
                images_count++;
            } else {
                printf("Failed to load image: %s.\n", filepath);
            }
        }
    }
    closedir(dir);

    printf("Loaded %d images.\n", images_count);

    DIR *out_dir = opendir(argv[2]);

    cudaEvent_t start_atom, stop_atom, start_priv, stop_priv;
    cudaEventCreate(&start_atom);
    cudaEventCreate(&stop_atom);
    cudaEventCreate(&start_priv);
    cudaEventCreate(&stop_priv);

    float elapsed_time_atom = 0.0f;
    float elapsed_time_priv = 0.0f;
    float time = 0.0f;

    unsigned int *h_histogram = (unsigned int *)malloc(256 * sizeof(unsigned int));

    // Process each image to compute histogram
    for (int i = 0; i < images_count; i++) {
        // Compute histogram
        int image_size = images[i].width * images[i].height * images[i].channels;
        size_t img_size = images[i].width * images[i].height * images[i].channels * sizeof(unsigned char);

        // Allocate device memory
        unsigned char *d_data;
        unsigned int *d_histogram;
        cudaMalloc((void **)&d_data, img_size);
        cudaMalloc((void **)&d_histogram, 256 * sizeof(unsigned int));

        // Initalize d_histogram to zero
        cudaMemset(d_histogram, 0, 256 * sizeof(unsigned int));

        // Copy image data to device
        cudaMemcpy(d_data, images[i].data, img_size, cudaMemcpyHostToDevice);

        // Atomic add kernel launch
        cudaEventRecord(start_atom);
        int block_size = 256;
        int num_blocks = (images[i].width * images[i].height * images[i].channels + block_size - 1) / block_size;
        compute_histogram_atomic_add<<<num_blocks, block_size>>>(d_data, d_histogram, image_size);
        cudaEventRecord(stop_atom);
        cudaEventSynchronize(stop_atom);
        cudaEventElapsedTime(&time, start_atom, stop_atom);
        elapsed_time_atom += time;

        // Privatization kernel launch
        cudaMemset(d_histogram, 0, 256 * sizeof(unsigned int));
        cudaEventRecord(start_priv);
        compute_histogram_priv<<<num_blocks, block_size>>>(d_data, d_histogram, image_size);
        cudaEventRecord(stop_priv);
        cudaEventSynchronize(stop_priv);
        cudaEventElapsedTime(&time, start_priv, stop_priv);
        elapsed_time_priv += time;

        // Copy histogram data to host
        cudaMemcpy(h_histogram, d_histogram, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaFree(d_data);
        cudaFree(d_histogram);

        // Note: Atomic add kernel output is checked against the CPU histogram
        // No need to allocate another host memory for it

        unsigned int *cpu_histogram = compute_histogram(images[i].data, images[i].width, images[i].height, images[i].channels);

        if (memcmp(cpu_histogram, h_histogram, 256 * sizeof(unsigned int)) == 0) {
            char filename[1024];
            snprintf(filename, sizeof(filename), "%s/%s_histogram.txt", argv[2], images[i].filename);

            FILE *file = fopen(filename, "w");

            for (int i = 0; i < 256; i++) {
                fprintf(file, "%d ", h_histogram[i]);
            }

            fclose(file);
        } else {
            printf("Histograms mismatch for %s\n", images[i].filename);
            break;
        }
        free(cpu_histogram);
    }
    closedir(out_dir);

    // Performance results
    printf("Total processing time for %d images using atomic add: %3f ms\n", images_count, elapsed_time_atom);
    printf("Average processing time per image using atomic add: %3f ms\n", elapsed_time_atom / images_count);
    printf("Total processing time for %d images using privatization: %3f ms\n", images_count, elapsed_time_priv);
    printf("Average processing time per image using privatization: %3f ms\n", elapsed_time_priv / images_count);

    // Free memory
    for (int i = 0; i < images_count; i++) {
        stbi_image_free(images[i].data);
    }

    free(h_histogram);
    cudaEventDestroy(start_atom);
    cudaEventDestroy(stop_atom);
    cudaEventDestroy(start_priv);
    cudaEventDestroy(stop_priv);

    return 0;
}