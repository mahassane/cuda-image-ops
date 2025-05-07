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


// Combined RGB Histogram
unsigned int* compute_histogram(unsigned char *data, int width, int height, int channels) {
    // Allocate histogram array
    // Calloc is used to initalized all elements to 0
    unsigned int *histogram = (unsigned int *)calloc(256, sizeof(unsigned int));

    // Loop over the three channels
    size_t data_length = (size_t)width * height * channels;
    for (size_t i = 0; i < data_length; i++) {
        histogram[data[i]]++;
    }

    return histogram;
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
                // printf("%d\n", images[images_count].width * images[images_count].height * images[images_count].channels);
                images_count++;
            } else {
                printf("Failed to load image: %s.\n", filepath);
            }
        }
    }
    closedir(dir);

    printf("Loaded %d images.\n", images_count);

    DIR *out_dir = opendir(argv[2]);

    clock_t start, end;

    // Process each image to compute histogram
    for (int i = 0; i < images_count; i++) {
        // Compute histogram
        start = clock();
        unsigned int *histogram = compute_histogram(images[i].data, images[i].width, images[i].height, images[i].channels);
        end = clock();
        double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        images[i].processing_time = cpu_time_used;

        char filename[1024];
        snprintf(filename, sizeof(filename), "%s/%s_histogram.txt", argv[2], images[i].filename);

        FILE *file = fopen(filename, "w");

        for (int i = 0; i < 256; i++) {
            fprintf(file, "%d ", histogram[i]);
        }

        fclose(file);

    }
    closedir(out_dir);

    double total_time = 0.0;

    for (int i = 0; i < images_count; i++) {
        total_time += images[i].processing_time;
    }
    printf("\nTotal processing time for %d images: %3f ms\n", images_count, total_time*1000.0);
    printf("Average processing time per image: %3f ms\n\n", (total_time/images_count)*1000.0);

    // Free memory
    for (int i = 0; i < images_count; i++) {
        stbi_image_free(images[i].data);
    }

    return 0;
}