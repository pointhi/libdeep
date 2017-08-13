/*
  Copyright (C) 2015-2017  Bob Mottram <bob@freedombone.net>

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

#include "tests_conv.h"

static void test_conv_init()
{
    int no_of_layers = 3;
    int image_width = 256;
    int image_height = 256;
    int image_depth = 3;
    int no_of_features = 10*10;
    int feature_width = 8;
    int final_image_width = 64;
    int final_image_height = 64;
    float match_threshold[] = { 0.0f, 0.0f, 0.0f };
    deeplearn_conv conv;

    printf("test_conv_init...");

    assert(conv_init(no_of_layers,
                     image_width, image_height, image_depth,
                     no_of_features, feature_width,
                     final_image_width, final_image_height,
                     &match_threshold[0], &conv) == 0);
    conv_free(&conv);

    printf("Ok\n");
}

static void test_conv_learn()
{
    int no_of_layers = 3;
    unsigned char * img = NULL;
    unsigned int image_width = 0;
    unsigned int image_height = 0;
    int image_depth = 3;
    int no_of_features = 4*4;
    int feature_width = 8;
    int final_image_width = 64;
    int final_image_height = 64;
    float match_threshold[] = { 0.0f, 0.0f, 0.0f };
    deeplearn_conv conv;
    unsigned int bitsperpixel = 0;
    unsigned int random_seed = 123;
    int downsampled_width=128;
    int downsampled_height=128;
    unsigned char downsampled[128*128];
    char filename[256];

    printf("test_conv_learn...");

    assert(deeplearn_read_png_file((char*)"Lenna.png",
                                   &image_width, &image_height,
                                   &bitsperpixel, &img)==0);
    assert(img != NULL);
    assert(image_width == 512);
    assert(image_height == 512);
    assert(bitsperpixel == 24);

    assert(image_resize(img, (int)image_width, (int)image_height,
                        (int)(bitsperpixel/8),
                        &downsampled[0],
                        downsampled_width, downsampled_height, 1)==0);
    image_depth = 1;
    image_width = (unsigned int)downsampled_width;
    image_height = (unsigned int)downsampled_height;

    assert(conv_init(no_of_layers,
                     (int)image_width, (int)image_height, image_depth,
                     no_of_features, feature_width,
                     final_image_width, final_image_height,
                     &match_threshold[0], &conv) == 0);

    float matching_score = 0;
    float prev_matching_score = 9999999;
    int error_decreases = 0;
    COUNTUP(i, 5) {
        matching_score =
            conv_learn(img, &conv, 100, &random_seed);
        assert(matching_score > 0);
        if (matching_score < prev_matching_score)
            error_decreases++;
        prev_matching_score = matching_score;
    }
    assert(error_decreases >= 4);

    /* force to the next layer */
    conv.match_threshold[0] = prev_matching_score+1000;

    /* check that error continues to decrease on the second layer */
    error_decreases = 0;
    prev_matching_score = 9999999;
    COUNTUP(i, 5) {
        matching_score =
            conv_learn(img, &conv, 100, &random_seed);
        assert(matching_score > 0);
        assert(conv.current_layer == 1);
        if (matching_score < prev_matching_score)
            error_decreases++;
        prev_matching_score = matching_score;
        printf(".");
        fflush(stdout);
    }
    assert(error_decreases >= 4);

    /* force to the next layer */
    conv.match_threshold[1] = prev_matching_score+1000;

    /* check that error continues to decrease on the second layer */
    error_decreases = 0;
    prev_matching_score = 9999999;
    COUNTUP(i, 5) {
        matching_score =
            conv_learn(img, &conv, 100, &random_seed);
        assert(matching_score > 0);
        assert(conv.current_layer == 2);
        if (matching_score < prev_matching_score)
            error_decreases++;
        prev_matching_score = matching_score;
        printf(".");
        fflush(stdout);
    }
    assert(error_decreases >= 4);

    /* clear outputs */
    FLOATCLEAR(&conv.outputs[0], conv.no_of_outputs);

    /* check that the outputs are all zero */
    float outputs_sum = 0;
    COUNTDOWN(i, conv.no_of_outputs) {
        assert(conv.outputs[i] == 0.0f);
    }

    /* feed forward through all layers */
    conv_feed_forward(&downsampled[0], &conv, no_of_layers);

    /* check that there are some non-zero outputs */
    outputs_sum = 0;
    COUNTDOWN(i, conv.no_of_outputs) {
        assert(conv.outputs[i] >= 0.0f);
        assert(conv.outputs[i] <= 1.0f);
        outputs_sum += conv.outputs[i];
    }
    outputs_sum /= conv.no_of_outputs;
    assert(outputs_sum > 0.01f);
    assert(outputs_sum <= 1.0f);

    /* save a graph */
    sprintf(filename,"%stemp_graph.png",DEEPLEARN_TEMP_DIRECTORY);
    conv_plot_history(&conv, filename, "Feature Learning",
                      1024, 480);

    free(img);
    conv_free(&conv);

    printf("Ok\n");
}

int run_tests_conv()
{
    printf("\nRunning convolution tests\n");

    test_conv_init();
    test_conv_learn();

    printf("All convolution tests completed\n");
    return 1;
}
