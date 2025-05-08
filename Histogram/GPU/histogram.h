extern unsigned int* compute_histogram(unsigned char *data, int width, int height, int channels) {
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