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
    double processing_time;
} Image;

unsigned char* apply_filter(unsigned char *input, int width, int height, int channels, float *mask, int mask_size) {
    int pad = mask_size / 2;
    // Allocate output memory
    unsigned char *output = (unsigned char *)malloc(width * height * channels);

    // Convolve each channel
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0.0f;

                for (int ky = -pad; ky <= pad; ky++) {
                    for (int kx = -pad; kx <= pad; kx++) {
                        int iy = row + ky;
                        int ix = col + kx;

                        if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                            int img_idx = (iy * width + ix) * channels + c;
                            int mask_idx = (ky + pad) * mask_size + (kx + pad);
                            sum += input[img_idx] * mask[mask_idx];
                        }
                    }
                }

                // Pixel value limitation
                if (sum < 0) sum = 0;
                if (sum > 255) sum = 255;

                // Returned image is RGB also
                int idx = (row * width + col) * channels + c;
                output[idx] = (unsigned char)sum;
            }
        }
    }

    return output;
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
    clock_t start, end;

    for (int i = 0; i < images_count; i++) {
        start = clock();
        unsigned char *filtered = apply_filter(images[i].data, images[i].width, images[i].height, images[i].channels, mask, mask_size);
        end = clock();
        images[i].processing_time = ((double)(end - start)) / CLOCKS_PER_SEC;

        char output_path[1024];
        snprintf(output_path, sizeof(output_path), "%s/filtered_%s", argv[2], images[i].filename);
        stbi_write_png(output_path, images[i].width, images[i].height, images[i].channels, filtered, images[i].width * images[i].channels);
        free(filtered);
    }

    double total_time = 0.0;
    for (int i = 0; i < images_count; i++) {
        total_time += images[i].processing_time;
    }

    printf("\nTotal processing time for %d images: %3f s\n", images_count, total_time);
    printf("Average processing time per image: %3f s\n\n", (total_time / images_count));

    for (int i = 0; i < images_count; i++) {
        stbi_image_free(images[i].data);
    }
    free(mask);

    return 0;
}
