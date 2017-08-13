/*
  libdeep - a library for deep learning
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

#ifndef DEEPLEARN_CONV_H
#define DEEPLEARN_CONV_H

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>
#include "globals.h"
#include "deeplearn_random.h"
#include "backprop_neuron.h"
#include "encoding.h"
#include "backprop.h"
#include "autocoder.h"
#include "deeplearn.h"
#include "deeplearn_features.h"
#include "deeplearn_pooling.h"

#define PREPROCESS_MAX_LAYERS 100
#define POOLING_FACTOR        2

typedef struct {
    int width, height, depth;
    float * layer;
    int no_of_features, feature_width;
    float * feature;
} deeplearn_conv_layer;

typedef struct {
    int no_of_layers;

    /* array storing layers */
    deeplearn_conv_layer layer[PREPROCESS_MAX_LAYERS];

    /* the outputs at the end of the process */
    int outputs_width;
    int no_of_outputs;
    float * outputs;

    float learning_rate;

    /* current layer for which features are being learned */
    int current_layer;

    /* minimum match score for each layer */
    float * match_threshold;

    /* training itterations elapsed */
    unsigned int itterations;

    /* training history */
    unsigned int history_plot_interval;
    char history_plot_filename[256];
    char history_plot_title[256];

    float history[DEEPLEARN_HISTORY_SIZE];
    int history_index, history_ctr, history_step;
} deeplearn_conv;

int conv_init(int no_of_layers,
              int image_width, int image_height, int image_depth,
              int no_of_features, int feature_width,
              int final_image_width, int final_image_height,
              float match_threshold[],
              deeplearn_conv * conv);

void conv_feed_forward(unsigned char * img, deeplearn_conv * conv, int layer);

float conv_learn(unsigned char * img,
                 deeplearn_conv * conv,
                 int samples,
                 unsigned int * random_seed);

void conv_free(deeplearn_conv * conv);

int conv_plot_history(deeplearn_conv * conv,
                      char * filename, char * title,
                      int img_width, int img_height);
int conv_save(FILE * fp, deeplearn_conv * conv);
int conv_load(FILE * fp, deeplearn_conv * conv);

void convolve_image(float img[],
                    int img_width, int img_height, int img_depth,
                    int feature_width, int no_of_features,
                    float feature[],
                    float layer[], int layer_width);

int conv_draw_features(unsigned char img[],
                       int img_width, int img_height, int img_depth,
                       int layer,
                       deeplearn_conv * conv);
int image_resize(unsigned char img[],
                 int image_width, int image_height, int image_depth,
                 unsigned char result[],
                 int result_width, int result_height, int result_depth);
void convolve_image_mono(float img[],
                         int img_width, int img_height,
                         int feature_width, int no_of_features,
                         float feature[],
                         float layer[], int layer_width);

#endif
