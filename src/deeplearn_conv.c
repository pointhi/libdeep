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
 * @param conv Instance to be updated
 * @param random_seed Random number generator seed
 * @returns zero on success
 */
int conv_init(int no_of_layers,
              int image_width, int image_height, int image_depth,
              int no_of_features, int feature_width,
              int final_image_width, int final_image_height,
              deeplearn_conv * conv,
              unsigned int * random_seed)
{
    conv->no_of_layers = no_of_layers;
    conv->current_layer = 0;

    conv->itterations = 0;
    conv->training_ctr = 0;

    COUNTUP(l, no_of_layers) {
        conv->layer[l].width =
            image_width - ((image_width-final_image_width)*l/(no_of_layers-1));
        conv->layer[l].height =
            image_height - ((image_height-final_image_height)*l/(no_of_layers-1));

        if (l == 0)
            conv->layer[l].depth = image_depth;
        else
            conv->layer[l].depth = conv->layer[l-1].no_of_features;

        conv->layer[l].no_of_features = no_of_features;
        conv->layer[l].feature_width =
            feature_width*conv->layer[l].width/image_width;
        if (conv->layer[l].feature_width < 3)
            conv->layer[l].feature_width = 3;

        /* allocate memory for arrays */
        conv->layer[l].layer =
            (float*)malloc(conv->layer[l].width*conv->layer[l].height*
                           conv->layer[l].depth*sizeof(float));
        if (!conv->layer[l].layer)
            return 1;

        conv->layer[l].feature =
            (float*)malloc(conv->layer[l].no_of_features*
                           conv->layer[l].feature_width*conv->layer[l].feature_width*
                           conv->layer[l].depth*sizeof(float));
        if (!conv->layer[l].feature)
            return 2;
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
    }
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
    /*
    if (FLOATWRITE(conv->reduction_factor) == 0)
        return -1;


    COUNTUP(i, conv->no_of_layers) {
        if (INTWRITE(conv->layer[i].pooling_factor) == 0)
            return -18;
    }
    */

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
    /*
    if (INTREAD(conv->reduction_factor) == 0)
        return -1;

    COUNTUP(i, conv->no_of_layers) {
        if (INTREAD(conv->layer[i].pooling_factor) == 0)
            return -20;
    }
    */

    return 0;
}
