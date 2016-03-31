/*
  Copyright (C) 2015  Bob Mottram <bob@robotics.uk.to>

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
    printf("test_conv_init...");

    unsigned int img_width = 512;
    unsigned int img_height = 512;
    unsigned int bitsperpixel = 24;
    int no_of_layers = 3;
    int max_features = 20;
    int reduction_factor = 4;
    int pooling_factor = 2;
    float error_threshold[] = {0.0, 0.0, 0.0};
    unsigned int random_seed = 648326;
    deeplearn_conv conv;

    assert(conv_init(no_of_layers,
                     img_width, img_height,
                     bitsperpixel/8, max_features,
                     reduction_factor, pooling_factor,
                     &conv, error_threshold,
                     &random_seed) == 0);
    conv_free(&conv);

    printf("Ok\n");
}

static void test_conv_image()
{
    printf("test_conv_image...");

    unsigned int img_width = 10;
    unsigned int img_height = 10;
    unsigned int bitsperpixel = 0;
    int no_of_layers = 3;
    int max_features = 20;
    int reduction_factor = 6;
    int pooling_factor = 2;
    float error_threshold[] = {0.0, 0.0, 0.0};
    unsigned int random_seed = 648326;
    unsigned char * img, * img2;
    deeplearn_conv conv;
    float BPerror = -1;
    char plot_filename[256];
    char plot_title[256];
    unsigned char use_dropouts = 0;

    /* load image from file */
    assert(deeplearn_read_png_file((char*)"Lenna.png",
                                   &img_width, &img_height,
                                   &bitsperpixel, &img)==0);

    img2 = (unsigned char*)malloc(128*128*3*sizeof(unsigned char));
    assert(img2);
    deeplearn_downsample(img, img_width, img_height,
                         img2, 128, 128);
    free(img);
    img = img2;
    img_width = 128;
    img_height = 128;

    assert(conv_init(no_of_layers,
                     img_width, img_height,
                     bitsperpixel/8, max_features,
                     reduction_factor, pooling_factor,
                     &conv, error_threshold,
                     &random_seed) == 0);

    int conv0_size =
        conv.layer[0].units_across *
        conv.layer[0].units_down * max_features;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < conv0_size; j++) {
            conv.layer[0].convolution[j] = -9999;
        }

        assert(conv_img(img, &conv, use_dropouts) == 0);

        /* check that some convolution happened */
        for (int j = 0; j < conv0_size; j++) {
            assert(conv.layer[0].convolution[j] != -9999);
        }

        /* error should be >= 0 */
        assert(conv.BPerror >= 0);
        /* error should be reducing */
        if (i > 0) {
            assert(conv.BPerror < BPerror);
        }
        BPerror = conv.BPerror;
    }

    /* move to the next layer */
    conv.BPerror = -1;
    conv.current_layer++;

    int conv1_size =
        conv.layer[1].units_across *
        conv.layer[1].units_down * max_features;

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < conv1_size; j++) {
            conv.layer[1].convolution[j] = -9999;
        }

        assert(conv_img(img, &conv, use_dropouts) == 0);

        /* check that some convolution happened */
        for (int j = 0; j < conv1_size; j++) {
            assert(conv.layer[1].convolution[j] != -9999);
        }

        /* error should be >= 0 */
        if (!(conv.BPerror >= 0)) {
            printf("\nBPerror: %f\n",conv.BPerror);
        }
        assert(conv.BPerror >= 0);
        /* error should be reducing */
        if (i > 0) {
            assert(conv.BPerror < BPerror);
        }
        BPerror = conv.BPerror;
    }

    sprintf(plot_filename,"/tmp/%s","libdeep_conv.png");
    sprintf(plot_title,"%s","Convolution Training Error");

    assert(conv_plot_history(&conv, plot_filename,
                             plot_title,
                             1024, 640) == 0);
    conv_free(&conv);
    free(img);

    printf("Ok\n");
}

static void test_conv_save_load()
{
    printf("test_conv_save_load...");

    unsigned int img_width = 512;
    unsigned int img_height = 512;
    unsigned int bitsperpixel = 24;
    int i, no_of_layers = 3;
    int max_features = 20;
    int reduction_factor = 4;
    int pooling_factor = 2;
    float error_threshold[] = {0.1, 0.2, 0.3};
    float error_threshold2[] = {0.6, 0.7, 0.8};
    unsigned int random_seed = 648326;
    deeplearn_conv conv1;
    deeplearn_conv conv2;
    FILE * fp;
    char filename[256];

    assert(conv_init(no_of_layers,
                     img_width, img_height,
                     bitsperpixel/8, max_features,
                     reduction_factor, pooling_factor,
                     &conv1, error_threshold,
                     &random_seed) == 0);

    sprintf(filename, "/tmp/%s", "libdeep_conv.dat");

    /* save */
    fp = fopen(filename,"w");
    assert(fp);
    assert(conv_save(fp, &conv1) == 0);
    fclose(fp);

    /* set some different values */
    conv2.reduction_factor = 45;
    conv2.pooling_factor = 8;
    conv2.inputs_across = 100;
    conv2.inputs_down = 200;
    conv2.inputs_depth = 16;
    conv2.no_of_layers = 2;
    conv2.max_features = 15;
    memcpy((void*)conv2.error_threshold,
           error_threshold2,
           conv2.no_of_layers*sizeof(float));
    conv2.random_seed = 20313;
    conv2.enable_learning = 0;
    conv2.current_layer = 4577;
    conv2.training_complete = 3;
    conv2.itterations = 642;
    conv2.layer[0].autocoder=NULL;

    /* load */
    fp = fopen(filename,"r");
    assert(fp);
    assert(conv_load(fp, &conv2) == 0);
    fclose(fp);

    /* compare the results */
    assert(conv2.layer[0].autocoder != NULL);
    assert(conv2.layer[0].autocoder->inputs != NULL);
    assert(conv2.layer[0].autocoder->hiddens != NULL);
    assert(conv1.layer[0].autocoder->NoOfInputs ==
           conv2.layer[0].autocoder->NoOfInputs);
    assert(conv1.layer[0].autocoder->NoOfHiddens ==
           conv2.layer[0].autocoder->NoOfHiddens);
    assert(conv1.reduction_factor == conv2.reduction_factor);
    assert(conv1.pooling_factor == conv2.pooling_factor);
    assert(conv1.random_seed != conv2.random_seed);
    assert(conv1.inputs_across == conv2.inputs_across);
    assert(conv1.inputs_down == conv2.inputs_down);
    assert(conv1.inputs_depth == conv2.inputs_depth);
    assert(conv1.max_features == conv2.max_features);
    assert(conv1.no_of_layers == conv2.no_of_layers);
    assert(conv1.enable_learning == conv2.enable_learning);
    assert(conv1.current_layer == conv2.current_layer);
    assert(conv1.training_complete == conv2.training_complete);
    assert(conv1.itterations == conv2.itterations);
    for (i = 0; i < conv1.no_of_layers; i++) {
        for (int j = 0; j < conv1.layer[i].autocoder->NoOfInputs*
                 conv1.layer[i].autocoder->NoOfHiddens; j++) {
            assert(conv1.layer[i].autocoder->weights[j] > -0.3f);
            assert(conv1.layer[i].autocoder->weights[j] < 0.3f);
            assert(conv2.layer[i].autocoder->weights[j] > -0.3f);
            assert(conv2.layer[i].autocoder->weights[j] < 0.3f);
        }
        assert(conv1.error_threshold[i] == conv2.error_threshold[i]);
        if ((conv1.layer[i].autocoder != NULL) &&
            (conv2.layer[i].autocoder != NULL)) {
            assert(autocoder_compare(conv1.layer[i].autocoder,
                                     conv2.layer[i].autocoder) == 0);
        }
        assert(conv1.layer[i].units_across == conv2.layer[i].units_across);
        assert(conv1.layer[i].units_down == conv2.layer[i].units_down);
        assert(conv1.layer[i].pooling_factor == conv2.layer[i].pooling_factor);
    }

    conv_free(&conv1);
    conv_free(&conv2);

    printf("Ok\n");
}

static void test_deconv_image()
{
    printf("test_deconv_image...");

    unsigned int img_width = 10;
    unsigned int img_height = 10;
    unsigned int bitsperpixel = 0;
    int no_of_layers = 3;
    int max_features = 20;
    int reduction_factor = 6;
    int pooling_factor = 2;
    float error_threshold[] = {0.0, 0.0, 0.0};
    unsigned int random_seed = 648326;
    unsigned char * img, * img2;
    deeplearn_conv conv;
    char conv_filename[256];
    unsigned char use_dropouts = 0;
    const int downsampled_width=128;
    int downsampled_size_bytes = downsampled_width*downsampled_width*3*sizeof(unsigned char);

    /* load image from file */
    assert(deeplearn_read_png_file((char*)"Lenna.png",
                                   &img_width, &img_height,
                                   &bitsperpixel, &img)==0);

    if (bitsperpixel != 24) {
        printf("unexpected image depth\n");
        free(img);
        assert(bitsperpixel == 24);
    }
    
    img2 = (unsigned char*)malloc(downsampled_size_bytes);
    assert(img2);
    deeplearn_downsample(img, img_width, img_height,
                         img2, downsampled_width, downsampled_width);
    free(img);
    img = img2;
    img_width = downsampled_width;
    img_height = downsampled_width;

    img2 = (unsigned char*)malloc(downsampled_size_bytes);
    assert(img2);

    assert(deeplearn_write_png_file("/tmp/test_deconv_image_original.png",
                                    downsampled_width, downsampled_width, 3*8, img)==0);

    assert(conv_init(no_of_layers,
                     img_width, img_height,
                     bitsperpixel/8, max_features,
                     reduction_factor, pooling_factor,
                     &conv, error_threshold,
                     &random_seed) == 0);

    int conv0_size =
        conv.layer[0].units_across *
        conv.layer[0].units_down * max_features;

    for (int step = 1; step < 5; step++) { 
        for (int i = 0; i < 50; i++) {
            /* prevent moving along from the first layer */
            for (int j = 0; j < conv0_size; j++) {
                conv.layer[0].convolution[j] = -9999;
            }

            assert(conv_img(img, &conv, use_dropouts) == 0);
        }
        assert(deconv_img(0, &conv, img2) == 0);
        sprintf(conv_filename,"/tmp/test_deconv_image_%d.png", step);
        assert(deeplearn_write_png_file(conv_filename,
                                        downsampled_width, downsampled_width, 3*8, img2)==0);
    }

    conv_free(&conv);
    free(img);
    free(img2);

    printf("Ok\n");
}

int run_tests_conv()
{
    printf("\nRunning convolution tests\n");

    test_conv_init();
    test_conv_image();
    test_conv_save_load();
    test_deconv_image();

    printf("All convolution tests completed\n");
    return 1;
}
