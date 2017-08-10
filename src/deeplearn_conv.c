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

#include "deeplearn_conv.h"

#define BEFORE_POOLING 0
#define AFTER_POOLING  1

/**
 * @brief Initialise a preprocessing system
 * @param no_of_layers The number of convolutional layers
 * @param inputs_across Number of units across in the input layer or image
 * @param inputs_down The number of units down in the input layer or image
 * @param inputs_depth The depth of the input layer or image
 * @param max_features The maximum number of features per layer
 * @param reduction_factor Reduction factor for successive convolution layers
 * @param pooling_factor The reduction factor used for pooling
 * @param conv Preprocessing object
 * @param random_seed Random number generator seed
 * @returns zero on success
 */
int conv_init(int no_of_layers,
              int inputs_across,
              int inputs_down,
              int inputs_depth,
              int max_features,
              int reduction_factor,
              int pooling_factor,
              deeplearn_conv * conv,
              float error_threshold[],
              unsigned int * random_seed)
{
    int across = inputs_across;
    int down = inputs_down;

    if (no_of_layers >= PREPROCESS_MAX_LAYERS)
        return -1;

    rand_num(random_seed);
    conv->random_seed = *random_seed;

    conv->reduction_factor = reduction_factor;
    conv->pooling_factor = pooling_factor;

    conv->history_ctr = 0;
    conv->training_ctr = 0;
    conv->history_index = 0;
    conv->history_step = 1;
    conv->history_plot_interval = 1;
    sprintf(conv->history_plot_filename,"%s","training_conv.png");
    sprintf(conv->history_plot_title,"%s",
            "Convolutional Training History");

    conv->current_layer = 0;
    conv->training_complete = 0;
    conv->itterations = 0;
    conv->BPerror = -1;
    memcpy((void*)conv->error_threshold,
           (void*)error_threshold, no_of_layers*sizeof(float));
    conv->enable_learning = 0;
    conv->no_of_layers = no_of_layers;
    conv->inputs_across = inputs_across;
    conv->inputs_down = inputs_down;
    conv->inputs_depth = inputs_depth;
    conv->max_features = max_features;

    COUNTUP(i, no_of_layers) {
        /* reduce the array dimensions */
        across /= reduction_factor;
        down /= reduction_factor;
        if (across < 4) across = 4;
        if (down < 4) down = 4;

        conv->layer[i].units_across = across;
        conv->layer[i].units_down = down;
        conv->layer[i].pooling_factor = pooling_factor;
        FLOATALLOC(conv->layer[i].convolution, across*down*conv_layer_features(conv, i));

        if (!conv->layer[i].convolution)
            return -2;

        /* ensure that the random seed is different for each
           convolutional neural net */
        rand_num(random_seed);

        /* create an autocoder for feature learning on this layer */
        conv->layer[i].autocoder = (ac*)malloc(sizeof(ac));

        int depth = conv_layer_features(conv, i);

        /* on the first layer the depth is the same as the
           inputs or image */
        if (i == 0)
            depth = inputs_depth;

        /* the number of units/pixels within an input patch of
           the previous layer, not including depth */
        int patch_pixels =
            conv_patch_radius(i,conv)*
            conv_patch_radius(i,conv)*4;

        /* initialise the autocoder for this layer */
        if (autocoder_init(conv->layer[i].autocoder,
                           patch_pixels*depth,
                           conv_layer_features(conv, i),
                           *random_seed) != 0)
            return -3;

        /* reduce the dimensions by the pooling factor */
        across /= pooling_factor;
        down /= pooling_factor;
        if (across < 4) across = 4;
        if (down < 4) down = 4;

        /* create a pooling array */
        FLOATALLOC(conv->layer[i].pooling, across*down*conv_layer_features(conv, i));
        if (!conv->layer[i].pooling)
            return -4;
    }
    return 0;
}

/**
 * @brief Frees memory for a preprocessing pipeline
 * @param conv Preprocessing object
 */
void conv_free(deeplearn_conv * conv)
{
    COUNTDOWN(i, conv->no_of_layers) {
        free(conv->layer[i].convolution);
        free(conv->layer[i].pooling);
        autocoder_free(conv->layer[i].autocoder);
        free(conv->layer[i].autocoder);
    }
}

/**
 * @brief Returns the number of features for the given convolution layer
 * @param conv Preprocessing object
 * @param layer_index Index of the layer
 * @return Number of features in the given layer
 */
int conv_layer_features(deeplearn_conv * conv, int layer_index)
{
    return conv->max_features -
        ((conv->max_features/2) * layer_index / conv->no_of_layers);
}

/**
 * @brief Update the learning history
 * @param conv Preprocessing object
 */
static void conv_update_history(deeplearn_conv * conv)
{
    int i;
    float error_value;

    if (conv->history_step == 0)
        return;

    conv->history_ctr++;
    if (conv->history_ctr >= conv->history_step) {
        error_value = conv->BPerror;
        if (error_value == DEEPLEARN_UNKNOWN_ERROR)
            error_value = 0;

        conv->history[conv->history_index] =
            error_value;
        conv->history_index++;
        conv->history_ctr = 0;

        /* show the learned features */
        /*
          if (conv->current_layer == 0) {
          features_plot_weights(conv->layer[0].autocoder,
          "learned_features.png",3,
          800, 800);
          }*/

        if (conv->history_index >= DEEPLEARN_HISTORY_SIZE) {
            for (i = 0; i < conv->history_index; i++)
                conv->history[i/2] = conv->history[i];

            conv->history_index /= 2;
            conv->history_step *= 2;
        }
    }
}

/**
 * @brief Returns the input layer patch radius for the given layer number
 * @param layer_index Index number of the convolution layer
 * @param conv Preprocessing object
 * @return Patch radius
 */
int conv_patch_radius(int layer_index,
                      deeplearn_conv * conv)
{
    int radius=0;

    if (layer_index == 0) {
        radius = conv->inputs_across/conv->layer[0].units_across;
    }
    else {
        int prev_pooling_factor =
            conv->layer[layer_index-1].pooling_factor;

        radius = (conv->layer[layer_index-1].units_across /
                  prev_pooling_factor) /
            conv->layer[layer_index].units_across;
    }

    if (radius < 2) radius = 2;
    return radius;
}

/**
 * @brief Returns the width of the layer
 * @param layer_index Index number of the convolution layer
 * @param conv Preprocessing object
 * @param after_pooling Whether to return the value before or after pooling
 * @return Layer width
 */
int conv_layer_width(int layer_index,
                     deeplearn_conv * conv,
                     int after_pooling)
{
    if (after_pooling == BEFORE_POOLING)
        return conv->layer[layer_index].units_across;

    return conv->layer[layer_index].units_across /
        conv->layer[layer_index].pooling_factor;
}

/**
 * @brief Returns the width of the final output
 * @param conv Preprocessing object
 * @return Width of the final output of the convolution system
 */
int conv_output_width(deeplearn_conv * conv)
{
    return conv_layer_width(conv->no_of_layers-1, conv, AFTER_POOLING);
}

/**
 * @brief Returns the height of the layer
 * @param layer_index Index number of the convolution layer
 * @param conv Preprocessing object
 * @param after_pooling Whether to return the value before or after pooling
 * @return Layer height
 */
int conv_layer_height(int layer_index,
                      deeplearn_conv * conv,
                      int after_pooling)
{
    if (after_pooling == BEFORE_POOLING)
        return conv->layer[layer_index].units_down;

    return conv->layer[layer_index].units_down /
        conv->layer[layer_index].pooling_factor;
}

/**
 * @brief Returns the width of the final output
 * @param conv Preprocessing object
 * @return Height of the final output of the convolution system
 */
int conv_output_height(deeplearn_conv * conv)
{
    return conv_layer_height(conv->no_of_layers-1, conv, AFTER_POOLING);
}

/**
 * @brief Returns the size of the final pooling array
 *        which is the ultimate output of the convolution system
 * @param conv Preprocessing object
 * @return Size of the final pooling array (number of floats)
 */
int conv_outputs(deeplearn_conv * conv)
{
    return conv_output_width(conv)*
        conv_output_height(conv)*
        conv_layer_features(conv, conv->no_of_layers-1);
}

/**
 * @brief Returns a value from the final pooling layer
 * @param conv Preprocessing object
 * @param index Index within the pooling array of the final convolution layer
 * @return Value from the final pooling layer
 */
float get_conv_output(deeplearn_conv * conv, int index)
{
    return conv->layer[conv->no_of_layers-1].pooling[index];
}

/**
 * @brief Returns the number of units in a convolution layer
 * @param layer_index Index number of the convolution layer
 * @param conv Preprocessing object
 * @return Number of units in the convolution layer
 */
int convolution_layer_units(int layer_index,
                            deeplearn_conv * conv)
{
    return conv->layer[layer_index].units_across*
        conv->layer[layer_index].units_down*
        conv_layer_features(conv, layer_index);
}

/**
 * @brief Convolution between the input image and the first layer
 * @param img Input image
 * @param conv Preprocessing object
 * @param BPerror Returned total backprop error from feature learning
 * @param use_dropouts Non-zero if dropouts are to be used
 * @return zero on success
 */
static int conv_img_initial(unsigned char img[],
                            deeplearn_conv * conv,
                            float * BPerror,
                            unsigned char use_dropouts)
{
    float currBPerror=0;
    int retval;
    int patch_radius = conv_patch_radius(0, conv);

    if (conv->enable_learning != 0) {
        /* do feature learning */
        retval =
            features_learn_from_img(conv_layer_width(0,conv,BEFORE_POOLING),
                                    conv_layer_height(0,conv,BEFORE_POOLING),
                                    patch_radius,
                                    conv->inputs_across,
                                    conv->inputs_down,
                                    conv->inputs_depth, img,
                                    convolution_layer_units(0, conv),
                                    conv->layer[0].autocoder,
                                    &currBPerror);

        if (retval != 0)
            return -1;

        *BPerror = *BPerror + currBPerror;
    }

    /* do the convolution for this layer */
    retval =
        features_conv_img_to_flt(conv_layer_width(0,conv,BEFORE_POOLING),
                                 conv_layer_height(0,conv,BEFORE_POOLING),
                                 patch_radius,
                                 conv->inputs_across,
                                 conv->inputs_down,
                                 conv->inputs_depth, img,
                                 convolution_layer_units(0,conv),
                                 conv->layer[0].convolution,
                                 conv->layer[0].autocoder,
                                 use_dropouts);
    if (retval != 0)
        return -2;

    return 0;
}

/**
 * @brief Subsequent convolution after the first layer
 * @param conv Preprocessing object
 * @param layer_index Index of the convolution layer (> 0)
 * @param BPerror Returned total backprop error from feature learning
 * @return zero on success
 */
static int conv_subsequent(deeplearn_conv * conv,
                           int layer_index,
                           float * BPerror)
{
    float currBPerror=0;
    int retval;
    int patch_radius = conv_patch_radius(layer_index, conv);
    unsigned char use_dropouts = 0;

    if (layer_index < 1)
        return -3;

    if (conv->enable_learning != 0) {
        /* do feature learning */
        retval =
            features_learn_from_flt(conv_layer_width(layer_index,conv,BEFORE_POOLING),
                                    conv_layer_height(layer_index,conv,BEFORE_POOLING),
                                    patch_radius,
                                    conv_layer_width(layer_index-1,conv,AFTER_POOLING),
                                    conv_layer_height(layer_index-1,conv,AFTER_POOLING),
                                    conv_layer_features(conv, layer_index),
                                    conv->layer[layer_index-1].pooling,
                                    convolution_layer_units(layer_index,conv),
                                    conv->layer[layer_index].autocoder,
                                    &currBPerror);

        if (retval != 0)
            return -4;

        *BPerror = *BPerror + currBPerror;
        use_dropouts = 1;
    }
    /* do the convolution for this layer */
    retval =
        features_conv_flt_to_flt(conv_layer_width(layer_index,conv,BEFORE_POOLING),
                                 conv_layer_height(layer_index,conv,BEFORE_POOLING),
                                 patch_radius,
                                 conv_layer_width(layer_index-1,conv,AFTER_POOLING),
                                 conv_layer_height(layer_index-1,conv,AFTER_POOLING),
                                 conv_layer_features(conv, layer_index),
                                 conv->layer[layer_index-1].pooling,
                                 convolution_layer_units(layer_index,conv),
                                 conv->layer[layer_index].convolution,
                                 conv->layer[layer_index].autocoder,
                                 use_dropouts);
    if (retval != 0)
        return -5;

    return 0;
}

/**
 * @brief Returns the current layer being trained
 * @param conv Preprocessing object
 * @return Current maximum layer
 */
static int get_max_layer(deeplearn_conv * conv)
{
    if (conv->training_complete == 0)
        return conv->current_layer+1;

    return conv->no_of_layers;
}

/**
 * @brief Enable or disable learning depending upon the given layer
 *        and training state
 * @param layer_index Index number of the convolution layer
 * @param conv Convolution object
 */
static void conv_enable_learning(int layer_index,
                                 deeplearn_conv * conv)
{
    int max_layer = get_max_layer(conv);

    if (conv->training_complete == 0) {
        /* enable learning on the current layer only */
        conv->enable_learning = 0;
        if (layer_index == max_layer-1)
            conv->enable_learning = 1;
    }
    else {
        /* NOTE: there could be some residual learning probability
           for use with online systems */
        conv->enable_learning = 0;
    }
}

/**
 * @brief Updates the current training error and moves to the next
 *        convolution layer if the error is low enough
 * @param layer_index The given layer index
 * @param BPerror Backpropogation error for this layer
 * @param conv Convolution object
 */
void conv_update_training_error(int layer_index,
                                float BPerror,
                                deeplearn_conv * conv)
{
    if (conv->training_complete != 0)
        return;

    int max_layer = get_max_layer(conv);
    if (layer_index < max_layer-1)
        return;

    if (conv->BPerror < 0) {
        conv->BPerror = BPerror;
        return;
    }

    /* update training error as a running average */
    conv->BPerror =
        (conv->BPerror*0.99f) + (BPerror*0.01f);

    /* record the history of error values */
    conv_update_history(conv);

    /* increment the number of itterations */
    if (conv->itterations < UINT_MAX)
        conv->itterations++;

    /* has the training for this layer been completed? */
    if (conv->BPerror <
        conv->error_threshold[layer_index]) {

        /* reset the error and move to the next layer */
        conv->BPerror = -1;
        conv->current_layer++;

        /* if this is the final layer */
        if (conv->current_layer >=
            conv->no_of_layers)
            conv->training_complete = 1;
    }
}

/**
 * @brief Performs deconvolution to an image as a series of
 *        deconvolutions and unpoolings
 * @param start_layer the convolution layer to beging from
 * @param conv Convolution object
 * @param img Input image which is the output
 * @returns zero on success
 */
int deconv_img(int start_layer,
               deeplearn_conv * conv,
               unsigned char img[])
{
    int max_layer = get_max_layer(conv);
    int img_width = conv->inputs_across;
    int img_height = conv->inputs_down;
    int img_depth = conv->inputs_depth;

    if (start_layer > max_layer-1) start_layer = max_layer-1;

    for (int layer_index = start_layer; layer_index > 0; layer_index--) {
        /* unpool the current layer */
        unpooling_from_flt_to_flt(conv_layer_features(conv, layer_index),
                                  conv_layer_width(layer_index,
                                                   conv, AFTER_POOLING),
                                  conv_layer_height(layer_index,
                                                    conv, AFTER_POOLING),
                                  conv->layer[layer_index].pooling,
                                  conv_layer_width(layer_index,
                                                   conv, BEFORE_POOLING),
                                  conv_layer_height(layer_index,
                                                    conv, BEFORE_POOLING),
                                  conv->layer[layer_index].convolution);

        /* deconvolve from the current layer to the pooling of the
           previous layer */
        features_deconv_flt_to_flt(
            conv_layer_width(layer_index, conv, BEFORE_POOLING),
            conv_layer_height(layer_index, conv, BEFORE_POOLING),
            conv_patch_radius(layer_index, conv),
            conv_layer_width(layer_index-1, conv, AFTER_POOLING),
            conv_layer_height(layer_index-1, conv, AFTER_POOLING),
            conv_layer_features(conv, layer_index),
            conv->layer[layer_index-1].pooling,
            convolution_layer_units(layer_index-1, conv),
            conv->layer[layer_index].convolution,
            conv->layer[layer_index-1].autocoder);
    }

    /* convert from the first layer back to the starting image */
    features_deconv_img_to_flt(conv_layer_width(0, conv, BEFORE_POOLING),
                               conv_layer_height(0, conv, BEFORE_POOLING),
                               conv_patch_radius(0, conv),
                               img_width, img_height, img_depth, img,
                               convolution_layer_units(0, conv),
                               conv->layer[0].convolution,
                               conv->layer[0].autocoder);
    return 0;
}

/**
 * @brief Performs convolution on an image as a series of
 *        convolutions and poolings
 * @param img Input image
 * @param conv Convolution object
 * @param use_dropouts Non-zero if dropouts are to be used
 * @returns zero on success
 */
int conv_img(unsigned char img[],
             deeplearn_conv * conv,
             unsigned char use_dropouts)
{
    int retval = -1;
    int max_layer = get_max_layer(conv);
    float BPerror;

    COUNTUP(i, max_layer) {
        conv_enable_learning(i, conv);

        BPerror = 0;

        if (i == 0)
            retval = conv_img_initial(img, conv, &BPerror, use_dropouts);
        else
            retval = conv_subsequent(conv, i, &BPerror);

        if (retval != 0)
            return retval;

        conv_update_training_error(i, BPerror, conv);

        /* pooling */
        retval =
            pooling_from_flt_to_flt(conv_layer_features(conv, i),
                                    conv_layer_width(i,conv,BEFORE_POOLING),
                                    conv_layer_height(i,conv,BEFORE_POOLING),
                                    conv->layer[i].convolution,
                                    conv_layer_width(i,conv,AFTER_POOLING),
                                    conv_layer_height(i,conv,AFTER_POOLING),
                                    conv->layer[i].pooling);
        if (retval != 0)
            return -6;
    }
    return 0;
}

/**
 * @brief Uses gnuplot to plot the training error
 * @param conv Convolution object
 * @param filename Filename for the image to save as
 * @param title Title of the graph
 * @param img_width Width of the image in pixels
 * @param img_height Height of the image in pixels
 * @return zero on success
 */
int conv_plot_history(deeplearn_conv * conv,
                      char * filename, char * title,
                      int img_width, int img_height)
{
    int retval=0;
    FILE * fp;
    char data_filename[256];
    char plot_filename[256];
    char command_str[256];
    float value;
    float max_value = 0.01f;

    sprintf(data_filename,"%s%s",DEEPLEARN_TEMP_DIRECTORY,
            "libdeep_conv_data.dat");
    sprintf(plot_filename,"%s%s",DEEPLEARN_TEMP_DIRECTORY,
            "libdeep_conv_data.plot");

    /* save the data */
    fp = fopen(data_filename,"w");

    if (!fp)
        return -1;

    COUNTUP(index, conv->history_index) {
        value = conv->history[index];
        fprintf(fp,"%d    %.10f\n",
                index*conv->history_step,value);
        /* record the maximum error value */
        if (value > max_value)
            max_value = value;
    }
    fclose(fp);

    /* create a plot file */
    fp = fopen(plot_filename,"w");

    if (!fp)
        return -1;

    fprintf(fp,"%s","reset\n");
    fprintf(fp,"set title \"%s\"\n",title);
    fprintf(fp,"set xrange [0:%d]\n",
            conv->history_index*conv->history_step);
    fprintf(fp,"set yrange [0:%f]\n",max_value*102/100);
    fprintf(fp,"%s","set lmargin 9\n");
    fprintf(fp,"%s","set rmargin 2\n");
    fprintf(fp,"%s","set xlabel \"Time Step\"\n");
    fprintf(fp,"%s","set ylabel \"Training Error Percent\"\n");

    fprintf(fp,"%s","set grid\n");
    fprintf(fp,"%s","set key right top\n");

    fprintf(fp,"set terminal png size %d,%d\n",
            img_width, img_height);
    fprintf(fp,"set output \"%s\"\n", filename);
    fprintf(fp,"plot \"%s\" using 1:2 notitle with lines\n",
            data_filename);
    fclose(fp);

    /* run gnuplot using the created files */
    sprintf(command_str,"gnuplot %s", plot_filename);
    retval = system(command_str); /* I assume this is synchronous */

    /* remove temporary files */
    sprintf(command_str,"rm %s %s", data_filename,plot_filename);
    retval = system(command_str);

    return retval;
}

/**
 * @brief Saves the given convolution object to a file
 * @param fp File pointer
 * @param conv Convolution object
 * @return zero value on success
 */
int conv_save(FILE * fp, deeplearn_conv * conv)
{
    if (FLOATWRITE(conv->reduction_factor) == 0)
        return -1;

    if (INTWRITE(conv->pooling_factor) == 0)
        return -2;

    if (UINTWRITE(conv->random_seed) == 0)
        return -3;

    if (INTWRITE(conv->inputs_across) == 0)
        return -4;

    if (INTWRITE(conv->inputs_down) == 0)
        return -5;

    if (INTWRITE(conv->inputs_depth) == 0)
        return -6;

    if (INTWRITE(conv->max_features) == 0)
        return -7;

    if (INTWRITE(conv->no_of_layers) == 0)
        return -8;

    if (BYTEWRITE(conv->enable_learning) == 0)
        return -9;

    if (INTWRITE(conv->current_layer) == 0)
        return -11;

    if (BYTEWRITE(conv->training_complete) == 0)
        return -12;

    if (UINTWRITE(conv->itterations) == 0)
        return -13;

    if (FLOATWRITEARRAY(conv->error_threshold, conv->no_of_layers) == 0)
        return -14;

    COUNTUP(i, conv->no_of_layers) {
        ac * net = conv->layer[i].autocoder;
        if (autocoder_save(fp, net) != 0)
            return -15;

        if (INTWRITE(conv->layer[i].units_across) == 0)
            return -16;

        if (INTWRITE(conv->layer[i].units_down) == 0)
            return -17;

        if (INTWRITE(conv->layer[i].pooling_factor) == 0)
            return -18;
    }

    return 0;
}

/**
 * @brief Loads a convolution object from file
 * @param fp File pointer
 * @param conv Convolution object
 * @return zero value on success
 */
int conv_load(FILE * fp, deeplearn_conv * conv)
{
    float * error_threshold;

    if (INTREAD(conv->reduction_factor) == 0)
        return -1;

    if (INTREAD(conv->pooling_factor) == 0)
        return -2;

    if (UINTREAD(conv->random_seed) == 0)
        return -3;

    if (INTREAD(conv->inputs_across) == 0)
        return -4;

    if (INTREAD(conv->inputs_down) == 0)
        return -5;

    if (INTREAD(conv->inputs_depth) == 0)
        return -6;

    if (INTREAD(conv->max_features) == 0)
        return -7;

    if (INTREAD(conv->no_of_layers) == 0)
        return -8;

    if (BYTEREAD(conv->enable_learning) == 0)
        return -9;

    if (INTREAD(conv->current_layer) == 0)
        return -11;

    if (BYTEREAD(conv->training_complete) == 0)
        return -12;

    if (UINTREAD(conv->itterations) == 0)
        return -13;

    FLOATALLOC(error_threshold, conv->no_of_layers);
    if (!error_threshold)
        return -14;

    if (FLOATREADARRAY(error_threshold, conv->no_of_layers) == 0)
        return -15;

    if (conv_init(conv->no_of_layers,
                  conv->inputs_across, conv->inputs_down,
                  conv->inputs_depth, conv->max_features,
                  conv->reduction_factor, conv->pooling_factor,
                  conv, error_threshold,
                  &conv->random_seed) != 0) {
        free(error_threshold);
        return -16;
    }
    free(error_threshold);

    COUNTUP(i, conv->no_of_layers) {
        if (autocoder_load(fp, conv->layer[i].autocoder, 0) != 0)
            return -17;

        if (INTREAD(conv->layer[i].units_across) == 0)
            return -18;

        if (INTREAD(conv->layer[i].units_down) == 0)
            return -19;

        if (INTREAD(conv->layer[i].pooling_factor) == 0)
            return -20;
    }

    return 0;
}

/**
 * @brief Sets the learning rate for the neural net at each convolution layer
 * @param conv Convolution object
 * @param rate the learning rate in the range 0.0 to 1.0
 */
void conv_set_learning_rate(deeplearn_conv * conv, float rate)
{
    COUNTDOWN(i, conv->no_of_layers)
        conv->layer[i].autocoder->learningRate = rate;
}

/**
 * @brief Sets the percentage of units which drop out during feature learning
 * @param conv Convolution object
 * @param dropout_percent Percentage of units which drop out in the range 0 to 100
 */
void conv_set_dropouts(deeplearn_conv * conv, float dropout_percent)
{
    COUNTDOWN(i, conv->no_of_layers)
        conv->layer[i].autocoder->DropoutPercent = dropout_percent;
}

/**
 * @brief Plots the features learned by an autocoder at the given layer
 * @param conv Convolution object
 * @param layer_index Index of the layer for which to plot the features
 * @param img Image array
 * @param img_width Width of the image
 * @param img_height Height of the image
 * @return zero on success
 */
int conv_plot_features(deeplearn_conv * conv, int layer_index,
                       unsigned char img[],
                       int img_width, int img_height)
{
    int patch_radius = conv_patch_radius(layer_index, conv);
    int patch_depth = conv_layer_features(conv, layer_index);
    int fx = (int)sqrt(conv_layer_features(conv, layer_index));
    int fy = conv_layer_features(conv, layer_index)/fx;
    ac * autocoder = conv->layer[layer_index].autocoder;

    if (layer_index >= conv->no_of_layers)
        return -100;

    /* clear the img with a white background */
    memset((void*)img, '\255', img_width*img_height*3*sizeof(unsigned char));

    if (layer_index == 0)
        patch_depth = conv->inputs_depth;

    /* for every feature within the autocoder */
    COUNTUP(y, fy) {
        int img_ty = y * img_height / fy;
        int img_by = img_ty + (img_height/fy) - 2;
        COUNTUP(x, fx) {
            int feature_index = (y*fx + x);
            if (feature_index >=
                conv_layer_features(conv, layer_index)) continue;
            int img_tx = x * img_width / fx;
            int img_bx = img_tx + (img_width/fx) - 2;

            int retval =
                autocoder_plot_weights(autocoder,
                                       feature_index,
                                       patch_radius, patch_depth,
                                       img_tx, img_ty, img_bx, img_by,
                                       img, img_width, img_height);
            if (retval != 0)
                return retval;
        }
    }
    return 0;
}
