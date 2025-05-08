#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include "../../utils/dirent.h"
#include <string.h>
#include "../../utils/stb_image.h"
#include "../../utils/stb_image_write.h"
#include <cuda_runtime.h>
#include <time.h>

typedef struct {
    unsigned char *data;
    int width;
    int height;
    int channels;
    char filename[1024];
} Image;

#define TILE_WIDTH 16
__constant__ float shared_mask[15 * 15]; // Maximum mask size is 15x15

__global__ void convolve_o_tiling(const unsigned char* input, unsigned char* output, int mask_size, int width, int height, int channels)
{
    int mask_radius = mask_size / 2;

    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    if (col < width && row < height) {
        // For each channel, do the convolution
        for (int c = 0; c < channels; ++c) {
            float pixel_value = 0.0f;

            // Loop over the mask
            for (int ky = -mask_radius; ky <= mask_radius; ++ky) {
                int iy = row + ky;
                if (iy < 0 || iy >= height) continue;

                for (int kx = -mask_radius; kx <= mask_radius; ++kx) {
                    int ix = col + kx;
                    if (ix < 0 || ix >= width) continue;

                    int img_idx  = (iy * width + ix) * channels + c;
                    int mask_idx = (ky + mask_radius) * mask_size + (kx + mask_radius);
                    pixel_value += input[img_idx] * shared_mask[mask_idx];
                }
            }

            // Clamp pixel value at the range [0-255]
            pixel_value = fminf(fmaxf(pixel_value, 0.0f), 255.0f);
            output[(row * width + col) * channels + c] = static_cast<unsigned char>(pixel_value);
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <input_directory> <output_directory> <mask_file>\n", argv[0]);
        return 1;
    }

    int mask_size;
    FILE *file = fopen(argv[3], "r");
    fscanf(file, "%d", &mask_size);
    float *mask = (float *)malloc(mask_size * mask_size * sizeof(float));

    for (int i = 0; i < mask_size * mask_size; i++) {
        fscanf(file, "%f", &mask[i]);
    }
    fclose(file);

    DIR *dir = opendir(argv[1]);

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsed_time = 0.0f;

    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    unsigned char *h_output[100];
    unsigned char *d_input[100];
    unsigned char *d_output[100];

    // Copy mask to constant memory
    cudaMemcpyToSymbol(shared_mask, mask, mask_size * mask_size * sizeof(float), 0, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    for (int i = 0; i < images_count; i++) {
        size_t img_size = images[i].width * images[i].height * images[i].channels * sizeof(unsigned char);

        int stream_id = i % 2;

        // Allocate pinned host memory
        unsigned char* pinned_input;
        cudaMallocHost((void **)&pinned_input, img_size);
        memcpy(pinned_input, images[i].data, img_size);
        cudaMallocHost((void **)&h_output[i], img_size);

        // Allocate device memory
        cudaMalloc((void **)&d_input[i], img_size);
        cudaMalloc((void **)&d_output[i], img_size);

        // Copy image data to device
        cudaMemcpyAsync(d_input[i], pinned_input, img_size, cudaMemcpyHostToDevice, stream[stream_id]);

        // Kernel launch
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((images[i].width + TILE_WIDTH - 1) / TILE_WIDTH, (images[i].height + TILE_WIDTH - 1) / TILE_WIDTH);
        convolve_o_tiling<<<gridDim, blockDim, 0, stream[stream_id]>>>(d_input[i], d_output[i], mask_size, images[i].width, images[i].height, images[i].channels);

        // Copy output data to host
        cudaMemcpyAsync(h_output[i], d_output[i], img_size, cudaMemcpyDeviceToHost, stream[stream_id]);
    }

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    for (int i = 0; i < images_count; i++) {    
        char output_path[1024];
        snprintf(output_path, sizeof(output_path), "%s/filtered_%s", argv[2], images[i].filename);
        stbi_write_png(output_path, images[i].width, images[i].height, images[i].channels, h_output[i], images[i].width * images[i].channels);
    
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
        cudaFreeHost(h_output[i]);
    }

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);

    // Performance results
    printf("Total processing time for %d images: %3f ms\n", images_count, elapsed_time);
    printf("Average processing time per image: %3f ms\n", elapsed_time / images_count);

    for (int i = 0; i < images_count; i++) {
        stbi_image_free(images[i].data);
    }
    free(mask);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
