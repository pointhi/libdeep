/*
 Feature learning from images demo
 Copyright (C) 2017  Bob Mottram <bob@freedombone.net>

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
 3. Neither the name of the University nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
 .
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE HOLDERS OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include "libdeep/globals.h"
#include "libdeep/deeplearn.h"

static void learn_features_from_image()
{
    unsigned int img_width = 0;
    unsigned int img_height = 0;
    unsigned int random_seed = 123;
    unsigned int bitsperpixel = 0;
    unsigned char * img, * img_features;
    float * img_float, * feature;
    int no_of_features = 16*16;
    int feature_width = 10;
    float * feature_score;
    int samples = 1000;
    int i;
    int features_img_width = 800;
    int features_img_height = 800;
    const float learning_rate = 0.1f;

    /* load image from file */
    assert(deeplearn_read_png_file((char*)"../../unittests/Lenna.png",
                                   &img_width, &img_height,
                                   &bitsperpixel, &img)==0);

    img_float = (float*)malloc(img_width*img_height*(bitsperpixel/8)*sizeof(float));
    if (!img_float) {
        printf("Failed to allocate image feature memory\n");
        free(img);
        return;
    }

    feature = (float*)malloc(no_of_features*feature_width*feature_width*(bitsperpixel/8)*sizeof(float));
    if (!feature) {
        printf("Failed to allocate learned feature memory\n");
        free(img_float);
        free(img);
        return;
    }
    feature_score = (float*)malloc(no_of_features*sizeof(float));
    if (!feature_score) {
        printf("Failed to allocate memory for feature scores\n");
        free(img_float);
        free(img);
        free(feature);
        return;
    }
    img_features =
        (unsigned char*)malloc(features_img_width*features_img_height*((int)bitsperpixel/8)*sizeof(unsigned char));
    if (!img_features) {
        printf("Failed to allocate memory for features image\n");
        free(img_float);
        free(img);
        free(feature);
        free(feature_score);
        return;
    }

    /* initially random features */
    for (i = 0; i < no_of_features*feature_width*feature_width*(bitsperpixel/8); i++)
        feature[i] = rand_num(&random_seed) % 256;

    /* convert the loaded image to floats */
    for (i = 0; i < img_width*img_height*(bitsperpixel/8); i++)
        img_float[i] = (float)img[i]/255.0f;

    for (i = 0; i < 30; i++) {
        float match_score =
            learn_features(img_float,
                           (int)img_width, (int)img_height, (int)bitsperpixel/8,
                           feature_width, no_of_features,
                           feature, feature_score,
                           samples, learning_rate, &random_seed);
        printf("%.4f\n", match_score);
    }

    printf("Learning completed\n");

    draw_features(img_features,
                  features_img_width, features_img_height,
                  (int)(bitsperpixel/8),
                  feature_width, no_of_features, feature);

    printf("test\n");

    deeplearn_write_png_file("features.png",
                             (unsigned int)features_img_width,
                             (unsigned int)features_img_height,
                             bitsperpixel, img_features);

    free(img_float);
    free(img);
    free(feature_score);
    free(feature);
    free(img_features);
}

int main(int argc, char* argv[])
{
    learn_features_from_image();
    return 0;
}
