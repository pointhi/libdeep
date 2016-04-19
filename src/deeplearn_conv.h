/*
  libdeep - a library for deep learning
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

typedef struct {
    ac * autocoder;
    int units_across, units_down;
    float * convolution;
    int pooling_factor;
    float * pooling;
} deeplearn_conv_layer;

typedef struct {
    int reduction_factor, pooling_factor;
    unsigned int random_seed;
    int inputs_across, inputs_down;
    int inputs_depth;
    int max_features;
    int no_of_layers;
    unsigned char enable_learning;
    deeplearn_conv_layer layer[PREPROCESS_MAX_LAYERS];

    /* keep track of training progress */
    int current_layer;
    float error_threshold[PREPROCESS_MAX_LAYERS];
    unsigned char training_complete;
    unsigned int itterations;
    float BPerror;

    /* training history */
    unsigned int training_ctr;
    unsigned int history_plot_interval;
    char history_plot_filename[256];
    char history_plot_title[256];

    float history[DEEPLEARN_HISTORY_SIZE];
    int history_index, history_ctr, history_step;
} deeplearn_conv;

int conv_patch_radius(int layer_index,
                      deeplearn_conv * conv);
int conv_layer_width(int layer_index,
                     deeplearn_conv * conv,
                     int after_pooling);
int conv_layer_height(int layer_index,
                      deeplearn_conv * conv,
                      int after_pooling);
int convolution_layer_units(int layer_index,
                            deeplearn_conv * conv);
int conv_init(int no_of_layers,
              int inputs_across,
              int inputs_down,
              int inputs_depth,
              int max_features,
              int reduction_factor,
              int pooling_factor,
              deeplearn_conv * conv,
              float error_threshold[],
              unsigned int * random_seed);

void conv_free(deeplearn_conv * conv);
int conv_img(unsigned char img[],
             deeplearn_conv * conv,
             unsigned char use_dropouts);
int deconv_img(int start_layer,
               deeplearn_conv * conv,
               unsigned char img[]);
int conv_plot_history(deeplearn_conv * conv,
                      char * filename, char * title,
                      int img_width, int img_height);
int conv_save(FILE * fp, deeplearn_conv * conv);
int conv_load(FILE * fp, deeplearn_conv * conv);
int conv_output_width(deeplearn_conv * conv);
int conv_output_height(deeplearn_conv * conv);
float get_conv_output(deeplearn_conv * conv, int index);
int conv_outputs(deeplearn_conv * conv);
void conv_set_learning_rate(deeplearn_conv * conv, float rate);
void conv_set_dropouts(deeplearn_conv * conv, float dropout_percent);
int conv_plot_features(deeplearn_conv * conv, int layer_index,
                       unsigned char img[],
                       int img_width, int img_height);
int conv_layer_features(deeplearn_conv * conv, int layer_index);
#endif
