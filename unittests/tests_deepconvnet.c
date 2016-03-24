/*
  Copyright (C) 2015-2016  Bob Mottram <bob@robotics.uk.to>

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

#include "tests_deepconvnet.h"

static void test_init()
{
    int no_of_convolutions = 2;
    int no_of_deep_layers = 2;
    int inputs_across = 320;
    int inputs_down = 240;
    int inputs_depth = 3;
    int max_features = 20;
    int reduction_factor = 2;
    int no_of_outputs = 10;
    deepconvnet convnet;
    float error_threshold[] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    unsigned int random_seed = 7423;

    printf("test_init...");

    assert(deepconvnet_init(no_of_convolutions,
                            no_of_deep_layers,
                            inputs_across,
                            inputs_down,
                            inputs_depth,
                            max_features,
                            reduction_factor,
                            no_of_outputs,
                            &convnet,
                            error_threshold,
                            &random_seed) == 0);

    deepconvnet_free(&convnet);

    printf("Ok\n");
}

static void set_pattern(unsigned char img[],
                        int img_width, int img_height,
                        int depth,
                        int pattern[])
{
    for (int y = 0; y < img_height; y++) {
        int yy = y * 7 / img_height;
        for (int x = 0; x < img_width; x++) {
            int xx = x * 7 / img_width;
            int n0 = (y*img_width + x) * depth;
            int n1 = yy*7 + xx;
            if (pattern[n1] != 0) {
                for (int c = 0; c < depth; c++) {
                    img[n0+c] = 255;
                }
            }
        }
    }
}

static void set_test_pattern(unsigned char img[],
                             int img_width, int img_height,
                             int depth, int shape)
{
    memset((void*)img,'\0',img_width*img_height*depth*sizeof(unsigned char));

    switch(shape%5) {
    case 0: {
        int pattern[] = {
            0,0,0,0,0,0,0,
            0,0,1,1,1,0,0,
            0,1,0,0,0,1,0,
            0,1,0,0,0,1,0,
            0,1,0,0,0,1,0,
            0,0,1,1,1,0,0,
            0,0,0,0,0,0,0
        };
        set_pattern(img, img_width, img_height, depth, pattern);
        break;
    }
    case 1: {
        int pattern[] = {
            0,0,0,0,0,0,0,
            0,0,0,1,0,0,0,
            0,0,0,1,0,0,0,
            0,1,1,1,1,1,0,
            0,1,0,1,0,0,0,
            0,0,0,1,0,0,0,
            0,0,0,0,0,0,0
        };
        set_pattern(img, img_width, img_height, depth, pattern);
        break;
    }
    case 2: {
        int pattern[] = {
            0,0,0,0,0,0,0,
            1,0,1,0,1,0,0,
            0,1,0,1,0,1,0,
            1,0,1,0,1,0,0,
            0,1,0,1,0,1,0,
            1,0,1,0,1,0,0,
            0,0,0,0,0,0,0
        };
        set_pattern(img, img_width, img_height, depth, pattern);
        break;
    }
    case 3: {
        int pattern[] = {
            0,0,0,0,0,0,0,
            0,1,1,1,1,1,0,
            0,1,0,0,0,1,0,
            0,1,0,0,0,1,0,
            0,1,0,0,0,1,0,
            0,1,1,1,1,1,0,
            0,0,0,0,0,0,0
        };
        set_pattern(img, img_width, img_height, depth, pattern);
        break;
    }
    case 4: {
        int pattern[] = {
            0,0,0,0,0,0,0,
            0,0,0,1,0,0,0,
            0,0,1,1,1,0,0,
            0,1,1,1,1,1,0,
            0,0,1,1,1,0,0,
            0,0,1,0,1,0,0,
            0,0,0,0,0,0,0
        };
        set_pattern(img, img_width, img_height, depth, pattern);
        break;
    }
    }
}

static void test_update_img()
{
    int i, no_of_convolutions = 2;
    int no_of_deep_layers = 2;
    int inputs_across = 320;
    int inputs_down = 240;
    int inputs_depth = 3;
    int max_features = 20;
    int reduction_factor = 2;
    int no_of_outputs = 10;
    deepconvnet convnet;
    float error_threshold[] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
    unsigned int random_seed = 7423;
    unsigned char * img =
        (unsigned char*)malloc(inputs_across*inputs_down*
                               inputs_depth*sizeof(unsigned char));
    float * conv_output =
        (float*)malloc(inputs_across*inputs_down*
                       inputs_depth*sizeof(float));
    float diff = 0, test_output[10];

    printf("test_update_img...");

    assert(deepconvnet_init(no_of_convolutions,
                            no_of_deep_layers,
                            inputs_across,
                            inputs_down,
                            inputs_depth,
                            max_features,
                            reduction_factor,
                            no_of_outputs,
                            &convnet,
                            error_threshold,
                            &random_seed) == 0);

    /* after training */
    convnet.convolution->training_complete = 1;
    convnet.learner->training_complete = 1;
    convnet.training_complete = 1;

    /* set some input value */
    set_test_pattern(img, inputs_across, inputs_down, inputs_depth, 1);

    /* clear the outputs */
    for (i = 0; i < no_of_outputs; i++) {
        deepconvnet_set_output(&convnet, i, 0.0f);
    }

    assert(deepconvnet_update_img(&convnet, img, -1)==0);

    for (i = 0; i < no_of_outputs; i++) {
        test_output[i] = deepconvnet_get_output(&convnet, i);
        assert(test_output[i] > 0.0f);
        assert(test_output[i] != 0.5f);
    }

    /* store the outputs of the convolution stage */
    for (i = 0; i < convnet.learner->net->NoOfInputs; i++) {
        conv_output[i] = convnet.convolution->layer[no_of_convolutions-1].pooling[i];
    }

    /* set some different input values */
    set_test_pattern(img, inputs_across, inputs_down, inputs_depth, 2);

    assert(deepconvnet_update_img(&convnet, img, -1)==0);

    /* check that the outputs of the convolution stage have changed */
    diff = 0;
    for (i = 0; i < convnet.learner->net->NoOfInputs; i++) {
        diff += fabs(conv_output[i] - convnet.convolution->layer[no_of_convolutions-1].pooling[i]);
    }
    assert(diff > 0);

    /* check that the outputs have changed */
    diff = 0;
    for (i = 0; i < no_of_outputs; i++) {
        diff += fabs(deepconvnet_get_output(&convnet, i) - test_output[i]);
    }
    assert(diff > 0);

    assert(deepconvnet_plot_features(&convnet, 0,
                                     "/tmp/features.png",
                                     640, 640) == 0);

    deepconvnet_free(&convnet);
    free(img);
    free(conv_output);

    printf("Ok\n");
}

static void test_learn_test_patterns()
{
    int no_of_convolutions = 4;
    int no_of_deep_layers = 2;
    int inputs_across = 32;
    int inputs_down = 32;
    int inputs_depth = 3;
    int max_features = 8;
    int reduction_factor = 2;
    int no_of_outputs = 5;
    deepconvnet convnet;
    float error_threshold[] = { 1.5, 7.0, 7.0, 7.0, 5.0, 5.0, 33.0 };
    float prev_error = DEEPLEARN_UNKNOWN_ERROR;
    int prev_layer = 0, ctr = 0, i, score = 0;
    unsigned int random_seed = 7423;
    unsigned char * img =
        (unsigned char*)malloc(inputs_across*inputs_down*
                               inputs_depth*sizeof(unsigned char));

    printf("test_learn_test_patterns...");

    assert(deepconvnet_init(no_of_convolutions,
                            no_of_deep_layers,
                            inputs_across,
                            inputs_down,
                            inputs_depth,
                            max_features,
                            reduction_factor,
                            no_of_outputs,
                            &convnet,
                            error_threshold,
                            &random_seed) == 0);

	convnet.history_plot_interval=200;
	deepconvnet_set_learning_rate(&convnet, 0.1f);

	int training_itteration = 0;
    while (convnet.training_complete == 0) {
        /* create the input image */
        set_test_pattern(img, inputs_across, inputs_down,
                         inputs_depth, training_itteration%no_of_outputs);

        /* learn */
        assert(deepconvnet_update_img(&convnet, img,
                                      training_itteration%no_of_outputs) == 0);

        /* check that a valid backprop error exists */
        if (training_itteration%30 == 0) {
            if (!(convnet.BPerror == DEEPLEARN_UNKNOWN_ERROR)) {
                if ((!(prev_error == DEEPLEARN_UNKNOWN_ERROR)) &&
					(prev_layer == convnet.current_layer)) {
                    if (prev_error <= convnet.BPerror) {
                        ctr++;
                        assert(ctr < 10);
                    }
                    else {
                        ctr = 0;
                    }
                }
                else {
                    ctr = 0;
                }
                prev_error = convnet.BPerror;
                prev_layer = convnet.current_layer;
            }
        }

        training_itteration++;
    }

    /* test the system */
    for (i = 0; i < no_of_outputs; i++) {
        /* input pattern */
        set_test_pattern(img, inputs_across, inputs_down,
                         inputs_depth, i);
        assert(deepconvnet_test_img(&convnet, img) == 0);
        if (deepconvnet_get_class(&convnet) == i) score++;
    }
    assert(score > 0);

    deepconvnet_free(&convnet);
    free(img);

    if (score < no_of_outputs) {
        printf("WARNING: score %d/%d\n", score, no_of_outputs);
    }
    else {
        printf("Ok\n");
    }
}

int run_tests_deepconvnet()
{
	printf("\nRunning deepconvnet tests\n");

	test_init();
	test_update_img();
	test_learn_test_patterns();

	printf("All deepconvnet tests completed\n");
	return 1;
}
