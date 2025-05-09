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
#define MASK_SIZE 15
#define MASK_RADIUS (MASK_SIZE / 2)

__constant__ float shared_mask[MASK_SIZE * MASK_SIZE];

__global__ void convolve_o_tiling(const unsigned char* input, unsigned char* output, int mask_size, int width, int height, int channels) {
    // Shared memory for producing one tile
    __shared__ unsigned char shared_input[TILE_WIDTH + 2 * MASK_RADIUS][TILE_WIDTH + 2 * MASK_RADIUS][3];

    int tile_dim = TILE_WIDTH + 2 * MASK_RADIUS;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Top left pixel of the tile in the original image
    int col_i = blockIdx.x * TILE_WIDTH;
    int row_i = blockIdx.y * TILE_WIDTH;

    // Load image into shared input
    for (int y = ty; y < tile_dim; y += blockDim.y) {
        for (int x = tx; x < tile_dim; x += blockDim.x) {
            int img_x = col_i + x - MASK_RADIUS;
            int img_y = row_i + y - MASK_RADIUS;
            for (int c = 0; c < channels; ++c) {
                if (img_x >= 0 && img_x < width && img_y >= 0 && img_y < height) {
                    shared_input[y][x][c] = input[(img_y * width + img_x) * channels + c];
                } else { // Padding
                    shared_input[y][x][c] = 0;
                }
            }
        }
    }
    __syncthreads();

    int col_o = col_i + tx;
    int row_o = row_i + ty;

    // Convolution
    if (col_o < width && row_o < height) {
        for (int c = 0; c < channels; ++c) {
            float pixel_value = 0.0f;

            for (int fy = -MASK_RADIUS; fy <= MASK_RADIUS; ++fy) {
                for (int fx = -MASK_RADIUS; fx <= MASK_RADIUS; ++fx) {
                    int sx = tx + fx + MASK_RADIUS;
                    int sy = ty + fy + MASK_RADIUS;
                    float in_val   = shared_input[sy][sx][c];
                    float mask_val = shared_mask[(fy + MASK_RADIUS)*mask_size + (fx + MASK_RADIUS)];
                    pixel_value += in_val * mask_val;
                }
            }

            // Pixel value range [0, 255]
            pixel_value = fminf(fmaxf(pixel_value, 0.0f), 255.0f);
            // Cast to unsigned char
            output[(row_o * width + col_o) * channels + c] = (unsigned char)pixel_value;
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

    while ((entry = readdir(dir)) != NULL && images_count < 80) {
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
    printf("Mask copied to constant memory.\n");


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
