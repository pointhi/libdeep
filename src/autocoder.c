/*
  libdeep - a library for deep learning
  Copyright (C) 2013-2017  Bob Mottram <bob@freedombone.net>

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

#include "autocoder.h"

/**
 * @brief Initialise an autocoder
 * @param autocoder Autocoder object
 * @param no_of_inputs The number of inputs
 * @param no_of_hiddens The number of hidden (encoder) units
 * @param random_seed Random number generator seed
 * @return zero on success
 */
int autocoder_init(ac * autocoder,
                   int no_of_inputs,
                   int no_of_hiddens,
                   unsigned int random_seed)
{
    autocoder->no_of_inputs = no_of_inputs;
    autocoder->no_of_hiddens = no_of_hiddens;

    FLOATALLOC(autocoder->inputs, no_of_inputs);
    if (!autocoder->inputs)
        return -1;

    FLOATALLOC(autocoder->hiddens, no_of_hiddens);
    if (!autocoder->hiddens)
        return -2;

    FLOATALLOC(autocoder->bias, no_of_hiddens);
    if (!autocoder->bias)
        return -3;

    FLOATALLOC(autocoder->weights, no_of_hiddens*no_of_inputs);
    if (!autocoder->weights)
        return -4;

    FLOATALLOC(autocoder->last_weight_change, no_of_hiddens*no_of_inputs);
    if (!autocoder->last_weight_change)
        return -5;

    FLOATALLOC(autocoder->outputs, no_of_inputs);
    if (!autocoder->outputs)
        return -6;

    FLOATALLOC(autocoder->bperr, no_of_hiddens);
    if (!autocoder->bperr)
        return -7;

    FLOATALLOC(autocoder->last_bias_change, no_of_hiddens);
    if (!autocoder->last_bias_change)
        return -8;

    FLOATCLEAR(autocoder->inputs, no_of_inputs);
    FLOATCLEAR(autocoder->outputs, no_of_inputs);
    FLOATCLEAR(autocoder->hiddens, no_of_hiddens);
    FLOATCLEAR(autocoder->last_weight_change, no_of_hiddens*no_of_inputs);
    FLOATCLEAR(autocoder->bperr, no_of_hiddens);
    FLOATCLEAR(autocoder->last_bias_change, no_of_hiddens);
    autocoder->backprop_error = AUTOCODER_UNKNOWN;
    autocoder->backprop_error_average = AUTOCODER_UNKNOWN;
    autocoder->learning_rate = 0.2f;
    autocoder->noise = 0;
    autocoder->random_seed = random_seed;
    autocoder->itterations = 0;
    autocoder->dropout_percent = 0.01f;

    /* initial small random values */
    COUNTDOWN(h, no_of_hiddens) {
        autocoder->bias[h] =
            rand_initial_weight(&autocoder->random_seed, 2);
        COUNTDOWN(i, no_of_inputs)
            autocoder->weights[h*no_of_inputs + i] =
                rand_initial_weight(&autocoder->random_seed, no_of_inputs);
    }
    return 0;
}

/**
 * @brief frees memory for an autocoder
 * @param autocoder Autocoder object
 */
void autocoder_free(ac * autocoder)
{
    free(autocoder->inputs);
    free(autocoder->outputs);
    free(autocoder->hiddens);
    free(autocoder->bias);
    free(autocoder->weights);
    free(autocoder->last_weight_change);
    free(autocoder->bperr);
    free(autocoder->last_bias_change);
}

/**
 * @brief Encodes the inputs to a given array
 * @param autocoder Autocoder object
 * @param encoded Array to store the encoded values
 * @param use_dropouts If non-zero then allow dropouts in the returned results
 */
void autocoder_encode(ac * autocoder, float * encoded,
                      unsigned char use_dropouts)
{
    COUNTDOWN(h, autocoder->no_of_hiddens) {
        if (use_dropouts != 0) {
            if (rand_num(&autocoder->random_seed)%10000 <
                autocoder->dropout_percent*100) {
                autocoder->hiddens[h] = (int)AUTOCODER_DROPPED_OUT;
                continue;
            }
        }

        /* weighted sum of inputs */
        float adder = autocoder->bias[h];
        float * w = &autocoder->weights[h*autocoder->no_of_inputs];
        float * inp = &autocoder->inputs[0];
        int i = autocoder->no_of_inputs-1;
        while (i >= 0) {
            adder += w[i] * inp[i];
            i--;
        }

        /* add some random noise */
        if (autocoder->noise > 0)
            adder = ((1.0f - autocoder->noise) * adder) +
                (autocoder->noise *
                 ((rand_num(&autocoder->random_seed)%10000)/10000.0f));

        /* activation function */
        encoded[h] = AF(adder);
    }
}

/**
 * @brief Decodes the encoded (hidden) units to a given output array
 * @param autocoder Autocoder object
 * @param decoded Array to store the decoded output values
 */
void autocoder_decode(ac * autocoder, float * decoded)
{
    COUNTDOWN(i, autocoder->no_of_inputs) {
        /* weighted sum of hidden inputs */
        float adder = 0;
        int h = autocoder->no_of_hiddens-1;
        float * w = &autocoder->weights[i];
        float * inp = &autocoder->hiddens[0];
        int step = autocoder->no_of_inputs;
        int ctr = h*step;
        while (h >= 0) {
            if (inp[h] != AUTOCODER_DROPPED_OUT)
                adder += w[ctr] * inp[h];

            ctr -= step;
            h--;
        }

        /* add some random noise */
        if (autocoder->noise > 0)
            adder = ((1.0f - autocoder->noise) * adder) +
                (autocoder->noise *
                 ((rand_num(&autocoder->random_seed)%10000)/10000.0f));

        /* activation function */
        decoded[i] = AF(adder);
    }
}

/**
 * @brief Feed forward
 * @param autocoder Autocoder object
 */
void autocoder_feed_forward(ac * autocoder)
{
    autocoder_encode(autocoder, autocoder->hiddens,1);
    autocoder_decode(autocoder, autocoder->outputs);
}

/**
 * @brief Back propogate the error
 * @param autocoder Autocoder object
 */
void autocoder_backprop(ac * autocoder)
{
    /* clear the backptop error for each hidden unit */
    FLOATCLEAR(autocoder->bperr, autocoder->no_of_hiddens);

    /* backprop from outputs to hiddens */
    autocoder->backprop_error = 0;
    float errorPercent = 0;
    COUNTDOWN(i, autocoder->no_of_inputs) {
        float backprop_error = autocoder->inputs[i] - autocoder->outputs[i];
        autocoder->backprop_error += fabs(backprop_error);
        errorPercent += fabs(backprop_error);
        float afact = autocoder->outputs[i] * (1.0f - autocoder->outputs[i]);
        float bperr = backprop_error * afact;
        float * w = &autocoder->weights[i];
        int h = autocoder->no_of_hiddens-1;
        int step = autocoder->no_of_inputs;
        int ctr = h*step;
        while (h >= 0) {
            if (autocoder->hiddens[h] != AUTOCODER_DROPPED_OUT)
                autocoder->bperr[h] += bperr * w[ctr];
            h--;
            ctr -= step;
        }
    }

    /* error percentage assuming an encoding range
       of 0.25 -> 0.75 */
    errorPercent = errorPercent * 100 / (0.6f*autocoder->no_of_inputs);

    /* update the running average */
    if (autocoder->backprop_error_average == AUTOCODER_UNKNOWN) {
        autocoder->backprop_error_average = autocoder->backprop_error;
        autocoder->backprop_error_percent = errorPercent;
    }
    else {
        autocoder->backprop_error_average =
            (autocoder->backprop_error_average*0.999f) +
            (autocoder->backprop_error*0.001f);
        autocoder->backprop_error_percent =
            (autocoder->backprop_error_percent*0.999f) +
            (errorPercent*0.001f);
    }

    /* increment the number of training itterations */
    if (autocoder->itterations < UINT_MAX)
        autocoder->itterations++;
}

/**
 * @brief Adjusts weights and biases
 * @param autocoder Autocoder object
 */
void autocoder_learn(ac * autocoder)
{
    /* weights between outputs and hiddens */
    float e = autocoder->learning_rate / (1.0f + autocoder->no_of_hiddens);
    COUNTDOWN(i, autocoder->no_of_inputs) {
        float afact = autocoder->outputs[i] * (1.0f - autocoder->outputs[i]);
        float backprop_error = autocoder->inputs[i] - autocoder->outputs[i];
        float gradient = afact * backprop_error;
        int step = autocoder->no_of_inputs;
        int n = (autocoder->no_of_hiddens-1)*step + i;
        COUNTDOWN(h, autocoder->no_of_hiddens) {
            if (autocoder->hiddens[h] != AUTOCODER_DROPPED_OUT) {
                autocoder->last_weight_change[n] =
                    e * (autocoder->last_weight_change[n] + 1) *
                    gradient * autocoder->hiddens[h];
                autocoder->weights[n] += autocoder->last_weight_change[n];
            }
            n -= step;
        }
    }

    /* weights between hiddens and inputs */
    e = autocoder->learning_rate / (1.0f + autocoder->no_of_inputs);
    COUNTDOWN(h, autocoder->no_of_hiddens) {
        if (autocoder->hiddens[h] == AUTOCODER_DROPPED_OUT)
            continue;

        float afact = autocoder->hiddens[h] * (1.0f - autocoder->hiddens[h]);
        float backprop_error = autocoder->bperr[h];
        float gradient = afact * backprop_error;
        autocoder->last_bias_change[h] =
            e * (autocoder->last_bias_change[h] + 1.0f) * gradient;
        autocoder->bias[h] += autocoder->last_bias_change[h];
        int n = (h+1)*autocoder->no_of_inputs - 1;
        COUNTDOWN(i, autocoder->no_of_inputs) {
            autocoder->last_weight_change[n] =
                e * (autocoder->last_weight_change[n] + 1) *
                gradient * autocoder->inputs[i];
            autocoder->weights[n] += autocoder->last_weight_change[n];
            n--;
        }
    }
}

/**
 * @brief Save an autocoder to file
 * @param fp Pointer to the file
 * @param autocoder Autocoder object
 * @return zero on success
 */
int autocoder_save(FILE * fp, ac * autocoder)
{
    if (INTWRITE(autocoder->no_of_inputs) == 0)
        return -1;

    if (INTWRITE(autocoder->no_of_hiddens) == 0)
        return -2;

    if (UINTWRITE(autocoder->random_seed) == 0)
        return -3;

    if (FLOATWRITE(autocoder->dropout_percent) == 0)
        return -4;

    if (FLOATWRITEARRAY(autocoder->weights,
                        autocoder->no_of_inputs*autocoder->no_of_hiddens) == 0)
        return -5;

    if (FLOATWRITEARRAY(autocoder->last_weight_change,
                        autocoder->no_of_inputs*autocoder->no_of_hiddens) == 0)
        return -6;

    if (FLOATWRITEARRAY(autocoder->bias, autocoder->no_of_hiddens) == 0)
        return -7;

    if (FLOATWRITEARRAY(autocoder->last_bias_change,
                        autocoder->no_of_hiddens) == 0)
        return -8;

    if (FLOATWRITE(autocoder->learning_rate) == 0)
        return -9;

    if (FLOATWRITE(autocoder->noise) == 0)
        return -10;

    if (UINTWRITE(autocoder->itterations) == 0)
        return -11;

    return 0;
}

/**
 * @brief Load an autocoder from file
 * @param fp Pointer to the file
 * @param autocoder Autocoder object
 * @param initialise Whether to initialise
 * @return zero on success
 */
int autocoder_load(FILE * fp, ac * autocoder, int initialise)
{
    int no_of_inputs = 0;
    int no_of_hiddens = 0;
    unsigned int random_seed = 0;

    if (INTREAD(no_of_inputs) == 0)
        return -1;

    if (INTREAD(no_of_hiddens) == 0)
        return -2;

    if (UINTREAD(random_seed) == 0)
        return -3;

    /* create the autocoder */
    if (initialise != 0) {
        if (autocoder_init(autocoder,
                           no_of_inputs,
                           no_of_hiddens,
                           random_seed) != 0) {
            return -4;
        }
    }
    else {
        autocoder->no_of_inputs = no_of_inputs;
        autocoder->no_of_hiddens = no_of_hiddens;
        autocoder->random_seed = random_seed;
    }

    if (FLOATREAD(autocoder->dropout_percent) == 0)
        return -5;

    if (FLOATREADARRAY(autocoder->weights,
                       no_of_inputs*no_of_hiddens) == 0)
        return -6;

    if (FLOATREADARRAY(autocoder->last_weight_change,
                       no_of_inputs*no_of_hiddens) == 0)
        return -7;

    if (FLOATREADARRAY(autocoder->bias, no_of_hiddens) == 0)
        return -8;

    if (FLOATREADARRAY(autocoder->last_bias_change, no_of_hiddens) == 0)
        return -9;

    if (FLOATREAD(autocoder->learning_rate) == 0)
        return -10;

    if (FLOATREAD(autocoder->noise) == 0)
        return -11;

    if (UINTREAD(autocoder->itterations) == 0)
        return -12;

    return 0;
}

/**
 * @brief Sets the input of an autocoder
 * @param autocoder Autocoder object
 * @param index Array index of the input
 * @param value The value to set the input to
 */
void autocoder_set_input(ac * autocoder, int index, float value)
{
    autocoder->inputs[index] = value;
}

/**
 * @brief Sets autocoder inputs from an array
 * @param autocoder Autocoder object
 * @param inputs Array containing input values
 */
void autocoder_set_inputs(ac * autocoder, float inputs[])
{
    memcpy((void*)autocoder->inputs, inputs,
           autocoder->no_of_inputs*sizeof(float));
}

/**
 * @brief Returns the value of a hidden unit
 * @param autocoder Autocoder object
 * @param index Array index of the hidden (encoder) unit
 * @return Value of the hidden (encoder) unit
 */
float autocoder_get_hidden(ac * autocoder, int index)
{
    return autocoder->hiddens[index];
}

/**
 * @brief Sets the value of a hidden unit
 * @param autocoder Autocoder object
 * @param index Array index of the hidden (encoder) unit
 * @param value Value to set as
 */
void autocoder_set_hidden(ac * autocoder, int index, float value)
{
    autocoder->hiddens[index] = value;
}

/**
 * @brief Main update routine for training
 * @param autocoder Autocoder object
 */
void autocoder_update(ac * autocoder)
{
    autocoder_feed_forward(autocoder);
    autocoder_backprop(autocoder);
    autocoder_learn(autocoder);
}

/**
 * @brief Normalises the inputs to the autocoder
 * @param autocoder Autocoder object
 */
void autocoder_normalise_inputs(ac * autocoder)
{
    float min = autocoder->inputs[0];
    float max = autocoder->inputs[0];

    FOR(i, 1, autocoder->no_of_inputs) {
        if (autocoder->inputs[i] < min)
            min = autocoder->inputs[i];

        if (autocoder->inputs[i] > max)
            max = autocoder->inputs[i];
    }

    float range = max - min;
    if (range <= 0) return;

    COUNTUP(i, autocoder->no_of_inputs) {
        autocoder->inputs[i] =
            0.25f + (((autocoder->inputs[i] - min)/range)*0.5f);
    }
}

/**
 * @brief Returns zero if two autocoders are the same
 * @param autocoder0 The first autocoder
 * @param autocoder1 The second autocoder
 * @return zero on success
 */
int autocoder_compare(ac * autocoder0, ac * autocoder1)
{
    if (autocoder0->no_of_inputs != autocoder1->no_of_inputs)
        return -1;

    if (autocoder0->no_of_hiddens != autocoder1->no_of_hiddens)
        return -2;

    COUNTDOWN(h, autocoder0->no_of_hiddens) {
        if (autocoder0->bias[h] != autocoder1->bias[h])
            return -3;
    }

    COUNTDOWN(i, autocoder0->no_of_inputs*autocoder0->no_of_hiddens) {
        if (autocoder0->weights[i] != autocoder1->weights[i])
            return -4;
    }
    return 0;
}

/**
 * @brief Plots weight values within an image
 * @param autocoder Autocoder object
 * @param feature_index Index number of the hidden (encoder) unit
 * @param patch_radius Radius of the patch in the input layer of a
 *        convolution system
 * @param patch depth Depth of the input layer of a convolution system
 * @param img_tx Top x coordinate for where to draw the weights
 * @param img_ty Top y coordinate for where to draw the weights
 * @param img_bx Bottom x coordinate for where to draw the weights
 * @param img_by Bottom y coordinate for where to draw the weights
 * @param img Image array (3 bytes per pixel)
 * @param img_width Width of the image
 * @param img_height Height of the image
 * @return zero on success
 */
int autocoder_plot_weights(ac * autocoder,
                           int feature_index,
                           int patch_radius, int patch_depth,
                           int img_tx, int img_ty, int img_bx, int img_by,
                           unsigned char img[],
                           int img_width, int img_height)
{
    int img_y_range = img_by - img_ty;
    int img_x_range = img_bx - img_tx;
    int patch_width = patch_radius*2;
    int no_of_weights = patch_width*patch_width*patch_depth;

    /* check that the number of inputs matches the expected patch size */
    if (autocoder->no_of_inputs != no_of_weights)
        return -1;

    float min_weight = autocoder->weights[0];
    float max_weight = min_weight;
    int start_index = feature_index*no_of_weights;
    FOR(i, start_index, start_index + no_of_weights) {
        if (autocoder->weights[i] < min_weight)
            min_weight = autocoder->weights[i];

        if (autocoder->weights[i] > max_weight)
            max_weight = autocoder->weights[i];
    }

    float weight_range = max_weight - min_weight;
    if (weight_range <= 0.0f) return -2;

    /* for every pixel in the output image */
    FOR(y, img_ty, img_by) {
        int patch_y = (y - img_ty) * patch_width / img_y_range;
        FOR(x, img_tx, img_bx) {
            int patch_x = (x - img_tx) * patch_width / img_x_range;

            /* position in the image */
            int img_n = (y*img_width + x)*3;

            /* position in the patch */
            int patch_n = (patch_y*patch_width + patch_x)*patch_depth;
            COUNTDOWN(c, 3) {
                float w = autocoder->weights[start_index + patch_n +
                                             (c*patch_depth/3)];
                img[img_n + c] =
                    (unsigned char)((w-min_weight)*255/weight_range);
            }
        }
    }
    return 0;
}

/**
* @brief Plots weight matrices within an image
* @param net Autocoder neural net object
* @param filename Filename of the image to save as
* @param image_width Width of the image in pixels
* @param image_height Height of the image in pixels
*/
int autocoder_plot_weight_matrix(ac * net,
                                 char * filename,
                                 int image_width, int image_height)
{
    float w, min_w=9999999.0f, max_w=-9999999.0f;
    float min_bias=9999999.0f, max_bias=-999999.0f;
    float min_hidden=999999.0f, max_hidden=-999999.0f;
    unsigned char * img;

    /* allocate memory for the image */
    img = (unsigned char*)malloc(image_width*image_height*3);
    if (!img)
        return -1;

    /* clear the image with a white background */
    memset((void*)img, '\255',
           image_width*image_height*3*sizeof(unsigned char));

    /* get the weight range */
    COUNTDOWN(h, net->no_of_hiddens) {
        COUNTDOWN(i, net->no_of_inputs) {
            w = net->weights[h*net->no_of_inputs + i];
            if (w < min_w) min_w = w;
            if (w > max_w) max_w = w;
        }
    }

    /* get the bias and hidden unit range */
    COUNTDOWN(h, net->no_of_hiddens) {
        if (net->bias[h] < min_bias) min_bias = net->bias[h];
        if (net->bias[h] > max_bias) max_bias = net->bias[h];
        if (net->hiddens[h] < min_hidden) min_hidden = net->hiddens[h];
        if (net->hiddens[h] > max_hidden) max_hidden = net->hiddens[h];
    }

    if (max_bias > min_bias) {
        COUNTDOWN(y, image_height) {
            int h = y*net->no_of_hiddens/image_height;
            COUNTDOWN(x, image_width) {
                int i = x*net->no_of_inputs/image_width;
                int n = (y*image_width + x)*3;
                w = net->weights[h*net->no_of_inputs + i];
                img[n] = (unsigned char)((w - min_w)*255/(max_w - min_w));
                img[n+1] =
                    (unsigned char)((net->bias[h]-min_bias)*255/
                                    (max_bias - min_bias));
                if (max_hidden > min_hidden)
                    img[n+2] =
                        (unsigned char)((net->hiddens[h]-min_hidden)*255/
                                        (max_hidden - min_hidden));
                else
                    img[n+2] = (unsigned char)255;
            }
        }
    }

    /* write the image to file */
    deeplearn_write_png_file(filename,
                             (unsigned int)image_width,
                             (unsigned int)image_height,
                             24, img);

    /* free the image memory */
    free(img);
    return 0;
}
