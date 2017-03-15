/*
 libdeep - a library for deep learning
 Copyright (C) 2016-2017  Bob Mottram <bob@freedombone.net>

 Differentiable Neural Computer (DNC)
 A neural Turing Machine architecture based on the paper:

   "Hybrid computing using a neural network with dynamic external memory"
   Nature, 2016

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

#ifndef DEEPLEARNDNC_H
#define DEEPLEARNDNC_H

#define DEEPLEARNDNC_READ_HEADS       2
#define DEEPLEARNDNC_WRITE_HEADS      1

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "globals.h"
#include "backprop.h"
#include "autocoder.h"
#include "encoding.h"
#include "deeplearndata.h"
#include "deeplearn.h"

struct deepl_dnc_memory {
    /* the number of addresses within the memory */
    unsigned int size;

    /* the width of each address */
    unsigned int width;

    /* The address space consisting of size * vectors with the given width */
    float ** address;

    /* memory recently used */
    float * usage;

    /* array used for key matching */
    float * similarity_score;

    /* address usage for each head
       dimension: size*size*(DEEPLEARNDNC_READ_HEADS + DEEPLEARNDNC_WRITE_HEADS) */
    float ** usage_temporal;

    /* the current addresses for each head */
    unsigned int address_ptr[DEEPLEARNDNC_READ_HEADS + DEEPLEARNDNC_WRITE_HEADS];
};
typedef struct deepl_dnc_memory deeplearn_dnc_memory;

struct deepl_dnc_read_head {
    float * key;
    char mode;
};
typedef struct deepl_dnc_read_head deeplearn_dnc_read_head;

struct deepl_dnc_write_head {
    float * key;
    float * write;
    float * erase;
};
typedef struct deepl_dnc_write_head deeplearn_dnc_write_head;

struct deepl_dnc {
    int no_of_inputs, no_of_outputs;
    deeplearn * controller;
    deeplearn_dnc_memory memory;
    deeplearn_dnc_read_head read_head[DEEPLEARNDNC_READ_HEADS];
    deeplearn_dnc_write_head write_head[DEEPLEARNDNC_WRITE_HEADS];
};
typedef struct deepl_dnc deeplearn_dnc;

int deeplearn_dnc_init(deeplearn_dnc * learner,
                       int memory_size, int memory_width,
                       int no_of_inputs,
                       int no_of_hiddens,
                       int hidden_layers,
                       int no_of_outputs,
                       float error_threshold[],
                       unsigned int * random_seed);
void deeplearn_dnc_feed_forward(deeplearn_dnc * learner);
void deeplearn_dnc_update(deeplearn_dnc * learner);
void deeplearn_dnc_free(deeplearn_dnc * learner);
void deeplearn_dnc_set_input_text(deeplearn_dnc * learner, char * text);
void deeplearn_dnc_set_input(deeplearn_dnc * learner, int index, float value);
int deeplearn_dnc_set_input_field(deeplearn_dnc * learner, int fieldindex, float value);
int deeplearn_dnc_set_input_field_text(deeplearn_dnc * learner, int fieldindex, char * text);
void deeplearn_dnc_set_inputs(deeplearn_dnc * learner, deeplearndata * sample);
void deeplearn_dnc_set_output(deeplearn_dnc * learner, int index, float value);
void deeplearn_dnc_set_outputs(deeplearn_dnc * learner, deeplearndata * sample);
void deeplearn_dnc_get_outputs(deeplearn_dnc * learner, float * outputs);
float deeplearn_dnc_get_output(deeplearn_dnc * learner, int index);
int deeplearn_dnc_get_class(deeplearn_dnc * learner);
void deeplearn_dnc_set_class(deeplearn_dnc * learner, int class);
int deeplearn_dnc_save(FILE * fp, deeplearn_dnc * learner);
int deeplearn_dnc_load(FILE * fp, deeplearn_dnc * learner,
                   unsigned int * random_seed);
int deeplearn_dnc_compare(deeplearn_dnc * learner1,
                      deeplearn_dnc * learner2);
int deeplearn_dnc_plot_history(deeplearn_dnc * learner,
                           char * filename, char * title,
                           int image_width, int image_height);
void deeplearn_dnc_inputs_from_image_patch(deeplearn_dnc * learner,
                                       unsigned char * img,
                                       int image_width, int image_height,
                                       int tx, int ty);
void deeplearn_dnc_inputs_from_image(deeplearn_dnc * learner,
                                 unsigned char * img,
                                 int image_width, int image_height);
void deeplearn_dnc_set_learning_rate(deeplearn_dnc * learner, float rate);
void deeplearn_dnc_set_dropouts(deeplearn_dnc * learner, float dropout_percent);
int deeplearn_dnc_export(deeplearn_dnc * learner, char * filename);
float deeplearn_dnc_get_error_threshold(deeplearn_dnc * learner, int index);
void deeplearn_dnc_set_error_threshold(deeplearn_dnc * learner, int index, float value);
void deeplearn_dnc_update_continuous(deeplearn_dnc * learner);
int deeplearn_dnc_training_last_layer(deeplearn_dnc * learner);
void deeplearn_dnc_update_read_heads(deeplearn_dnc * learner);
void deeplearn_dnc_update_write_heads(deeplearn_dnc * learner);
void deeplearn_dnc_memory_update(unsigned int previous_address,
                                 unsigned int current_address,
                                 deeplearn_dnc_memory * memory,
                                 unsigned char write);
void deeplearn_dnc_update_similarity_scores(unsigned int current_address,
                                            float * key,
                                            deeplearn_dnc_memory * memory,
                                            unsigned char forward);

#endif
