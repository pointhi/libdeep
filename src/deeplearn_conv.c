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

/**
 * @brief Create a number of convolutional layers
 * @param no_of_layers The number of layers
 * @param image_width Width of the input image or layer
 * @param image_height Height of the input image or layer
 * @param image_depth Depth of the input image
 * @param no_of_features The number of features to learn in the first layer
 * @param feature_width Width of features in the first layer
 * @param final_image_width Width of the final output layer
 * @param final_image_height Height of the final layer
 * @param match_threshold Array containing the minimum matching threshold
 *        for each convolution layer
 * @param conv Instance to be updated
 * @returns zero on success
 */
int conv_init(int no_of_layers,
              int image_width, int image_height, int image_depth,
              int no_of_features, int feature_width,
              int final_image_width, int final_image_height,
              float match_threshold[],
              deeplearn_conv * conv)
{
    /* used to initially randomize the learned feature arrays */
    unsigned int rand_seed = 234;

    conv->no_of_layers = no_of_layers;
    conv->current_layer = 0;
    conv->learning_rate = 0.1f;

    conv->itterations = 0;
    conv->history_ctr = 0;
    conv->history_index = 0;
    conv->history_step = 1;

    conv->history_plot_interval = 10;
    sprintf(conv->history_plot_filename,"%s","feature_learning.png");
    sprintf(conv->history_plot_title,"%s","Feature Learning Training History");

    COUNTUP(l, no_of_layers) {
        conv->layer[l].width =
            image_width - ((image_width-final_image_width)*l/no_of_layers);

        /* After the initial layer, width and height are the same */
        if (l == 0)
            conv->layer[l].height =
                image_height - ((image_height-final_image_height)*l/no_of_layers);
        else
            conv->layer[l].height = conv->layer[l].width;

        /* first layer is the image depth, after that depth is the number
           of features on the previous layer */
        if (l == 0)
            conv->layer[l].depth = image_depth;
        else
            conv->layer[l].depth = conv->layer[l-1].no_of_features;

        conv->layer[l].no_of_features = no_of_features;

        /* make feature width proportional to width of the layer */
        conv->layer[l].feature_width =
            feature_width*conv->layer[l].width/image_width;

        /* feature width should not be too small */
        if (conv->layer[l].feature_width < 3)
            conv->layer[l].feature_width = 3;

        /* allocate memory for the layer */
        FLOATALLOC(conv->layer[l].layer,
                   conv->layer[l].width*conv->layer[l].height*
                   conv->layer[l].depth);
        if (!conv->layer[l].layer)
            return 1;
        FLOATCLEAR(conv->layer[l].layer,
                   conv->layer[l].width*conv->layer[l].height*
                   conv->layer[l].depth);

        /* allocate memory for learned feature set */
        FLOATALLOC(conv->layer[l].feature,
                   conv->layer[l].no_of_features*
                   conv->layer[l].feature_width*conv->layer[l].feature_width*
                   conv->layer[l].depth);
        if (!conv->layer[l].feature)
            return 2;
        COUNTDOWN(r, conv->layer[l].no_of_features*
                  conv->layer[l].feature_width*conv->layer[l].feature_width*
                  conv->layer[l].depth)
            conv->layer[l].feature[r] = (rand_num(&rand_seed) % 10000)/10000.0f;
    }

    conv->outputs_width = final_image_width;

    /* for convenience this is the size of the outputs array */
    conv->no_of_outputs =
        final_image_width*final_image_width*conv->layer[no_of_layers-1].depth;

    /* allocate array of output values */
    FLOATALLOC(conv->outputs, conv->no_of_outputs);
    if (!conv->outputs)
        return 3;

    /* allocate array containing training thresholds */
    FLOATALLOC(conv->match_threshold, conv->no_of_layers);
    if (!conv->match_threshold)
        return 4;

    /* copy threshold values into the array */
    memcpy((void*)conv->match_threshold, match_threshold,conv->no_of_layers*sizeof(float));

    return 0;
}

/**
 * @brief Frees memory for a preprocessing pipeline
 * @param conv Preprocessing object
 */
void conv_free(deeplearn_conv * conv)
{
    COUNTDOWN(l, conv->no_of_layers) {
        free(conv->layer[l].layer);
        free(conv->layer[l].feature);
    }

    free(conv->outputs);
    free(conv->match_threshold);
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
    if (FLOATWRITE(conv->no_of_layers) == 0)
        return -1;

    COUNTUP(l, conv->no_of_layers) {
        if (INTWRITE(conv->layer[l].width) == 0)
            return -2;
        if (INTWRITE(conv->layer[l].height) == 0)
            return -3;
        if (INTWRITE(conv->layer[l].depth) == 0)
            return -4;
        if (INTWRITE(conv->layer[l].no_of_features) == 0)
            return -5;
        if (INTWRITE(conv->layer[l].feature_width) == 0)
            return -6;
    }

    if (INTWRITE(conv->outputs_width) == 0)
        return -7;
    if (INTWRITE(conv->no_of_outputs) == 0)
        return -8;
    if (INTWRITE(conv->learning_rate) == 0)
        return -9;
    if (INTWRITE(conv->current_layer) == 0)
        return -10;
    if (FLOATWRITEARRAY(conv->match_threshold,
                        conv->no_of_layers) == 0)
        return -11;
    if (UINTWRITE(conv->itterations) == 0)
        return -12;

    /* save the history */
    if (INTWRITE(conv->history_index) == 0)
        return -13;

    if (INTWRITE(conv->history_ctr) == 0)
        return -14;

    if (INTWRITE(conv->history_step) == 0)
        return -15;

    if (FLOATWRITEARRAY(conv->history,
                        conv->history_index) == 0)
        return -16;

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
    if (FLOATREAD(conv->no_of_layers) == 0)
        return -1;

    COUNTUP(l, conv->no_of_layers) {
        if (INTREAD(conv->layer[l].width) == 0)
            return -2;
        if (INTREAD(conv->layer[l].height) == 0)
            return -3;
        if (INTREAD(conv->layer[l].depth) == 0)
            return -4;
        if (INTREAD(conv->layer[l].no_of_features) == 0)
            return -5;
        if (INTREAD(conv->layer[l].feature_width) == 0)
            return -6;
    }

    if (INTREAD(conv->outputs_width) == 0)
        return -7;
    if (INTREAD(conv->no_of_outputs) == 0)
        return -8;

    float match_threshold[PREPROCESS_MAX_LAYERS];
    conv_init(conv->no_of_layers,
              conv->layer[0].width, conv->layer[0].height, conv->layer[0].depth,
              conv->layer[0].no_of_features, conv->layer[0].feature_width,
              conv->outputs_width, conv->outputs_width,
              &match_threshold[0], conv);

    if (INTREAD(conv->learning_rate) == 0)
        return -9;
    if (INTREAD(conv->current_layer) == 0)
        return -10;
    if (FLOATREADARRAY(conv->match_threshold,
                        conv->no_of_layers) == 0)
        return -11;
    if (UINTREAD(conv->itterations) == 0)
        return -12;

    /* load the history */
    if (INTREAD(conv->history_index) == 0)
        return -13;

    if (INTREAD(conv->history_ctr) == 0)
        return -14;

    if (INTREAD(conv->history_step) == 0)
        return -15;

    if (FLOATREADARRAY(conv->history, conv->history_index) == 0)
        return -16;

    return 0;
}

/**
 * @brief Convolves an input image or layer to an output layer
 * @param img Input image or previous layer with values in the range 0.0 -> 1.0
 * @param img_width Width of the image
 * @param img_height Height of the image
 * @param img_depth Depth of the image. If this is the first layer then it is
 *        the color depth, otherwise it is the number of features learned in
 *        the previous layer
 * @param feature_width Width if each image patch
 * @param no_of_features The number of features in the set
 * @param feature Array containing the learned features, having values in
 *        the range 0.0 -> 1.0
 * @param layer The output layer
 * @param layer_width Width of the output layer. The total size of the
 *        output layer should be layer_width*layer_width*no_of_features
 */
void convolve_image(float img[],
                    int img_width, int img_height, int img_depth,
                    int feature_width, int no_of_features,
                    float feature[],
                    float layer[], int layer_width)
{
    float feature_pixels =
        1.0f / (float)(feature_width*feature_width*img_depth);

    COUNTDOWN(layer_y, layer_width) {
        int ty = layer_y * img_height / layer_width;
        int by = (layer_y+1) * img_height / layer_width;
        COUNTDOWN(layer_x, layer_width) {
            int tx = layer_x * img_width / layer_width;
            int bx = (layer_x+1) * img_width / layer_width;
            COUNTDOWN(f, no_of_features) {
                float * curr_feature =
                    &feature[f*feature_width*feature_width*img_depth];

                float match = 0.0f;
                COUNTDOWN(yy, feature_width) {
                    int tyy = ty + (yy * (by-ty) / feature_width);
                    COUNTDOWN(xx, feature_width) {
                        int txx = tx + (xx * (bx-tx) / feature_width);
                        int n0 = ((tyy*img_width) + txx) * img_depth;
                        int n1 = ((yy * feature_width) + xx) * img_depth;
                        COUNTDOWN(d, img_depth)
                            match +=
                                (img[n0+d] - curr_feature[n1+d])*
                                (img[n0+d] - curr_feature[n1+d]);
                    }
                }

                layer[((layer_y*layer_width) + layer_x)*no_of_features + f] =
                    1.0f - (float)sqrt(match * feature_pixels);
            }
        }
    }
}

/**
 * @brief Feed forward to the given layer
 * @param img The input image
 * @param conv Convolution instance
 * @param layer The number of layers to convolve
 */
void conv_feed_forward(unsigned char * img,
                       deeplearn_conv * conv, int layer)
{
    /* convert the input image to floats */
    COUNTDOWN(i, conv->layer[0].width*conv->layer[0].height*conv->layer[0].depth)
        conv->layer[0].layer[i] = (float)img[i]/255.0f;

    COUNTUP(l, layer) {
        float * next_layer = conv->outputs;
        int next_layer_width = conv->outputs_width;

        if (l < conv->no_of_layers-1) {
            next_layer = conv->layer[l+1].layer;
            next_layer_width = conv->layer[l+1].width;
        }

        convolve_image(conv->layer[l].layer,
                       conv->layer[l].width, conv->layer[l].height,
                       conv->layer[l].depth,
                       conv->layer[l].feature_width,
                       conv->layer[l].no_of_features,
                       conv->layer[l].feature,
                       next_layer, next_layer_width);
    }
}

/**
 * @brief Update the history of scores during feature learning
 * @param conv Convolution instance
 * @param matching score Current score when matching features
 */
static void conv_update_history(deeplearn_conv * conv,
                                float matching_score)
{
    conv->itterations++;

    if (conv->history_step == 0) return;

    conv->history_ctr++;
    if (conv->history_ctr >= conv->history_step) {

        conv->history[conv->history_index] =
            matching_score;
        conv->history_index++;
        conv->history_ctr = 0;

        if (conv->history_index >= DEEPLEARN_HISTORY_SIZE) {
            COUNTUP(i, conv->history_index)
                conv->history[i/2] = conv->history[i];

            conv->history_index /= 2;
            conv->history_step *= 2;
        }
    }
}

/**
 * @brief Learn features
 * @param conv Convolution instance
 * @param samples The number of samples from the image or layer
 * @param random_seed Random number generator seed
 * @returns matching score/error, with lower values being better match
 */
float conv_learn(unsigned char * img,
                 deeplearn_conv * conv,
                 int samples,
                 unsigned int * random_seed)
{
    float matching_score = 0;
    float * feature_score;
    int layer = conv->current_layer;

    if (layer >= conv->no_of_layers)
        return 0;

    FLOATALLOC(feature_score, conv->layer[layer].no_of_features);

    if (!feature_score)
        return -1;

    conv_feed_forward(img, conv, layer);

    matching_score +=
        learn_features(&conv->layer[layer].layer[0],
                       conv->layer[layer].width,
                       conv->layer[layer].height,
                       conv->layer[layer].depth,
                       conv->layer[layer].feature_width,
                       conv->layer[layer].no_of_features,
                       &conv->layer[layer].feature[0],
                       feature_score,
                       samples, conv->learning_rate, random_seed);
    /* check for NaN */
    if (matching_score != matching_score) {
        printf("matching_score = %f\n", matching_score);
        free(feature_score);
        return -2;
    }

    conv_update_history(conv, matching_score);

    free(feature_score);

    /* proceed to the next layer if the match is good enough */
    if (matching_score < conv->match_threshold[layer])
        conv->current_layer++;

    return matching_score;
}

/**
 * @brief Draws features for a given convolution layer
 * @param img Image to draw to
 * @param img_width Width of the image
 * @param img_height Height of the image
 * @param img_depth Depth of the image
 * @param layer Index of the layer whose features will be shown
 * @param conv Convolution instance
 * @returns zero on success
 */
int conv_draw_features(unsigned char img[],
                       int img_width, int img_height, int img_depth,
                       int layer,
                       deeplearn_conv * conv)
{
    int feature_width;
    int no_of_features;
    float * feature;

    if ((layer < 0) || (layer >= conv->no_of_layers))
        return -1;

    feature_width = conv->layer[layer].feature_width;
    no_of_features = conv->layer[layer].no_of_features;
    feature = conv->layer[layer].feature;

    if (layer == 0)
        return draw_features(img, img_width, img_height, img_depth,
                             feature_width, no_of_features, feature);

    /* TODO: subsequent layers */

    return 0;
}
