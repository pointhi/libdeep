/*
  libdeep - a library for deep learning
  Copyright (C) 2015-2016  Bob Mottram <bob@robotics.uk.to>

  A deep convolution network combines a convolutional net as a
  preprocessing stage together with a deep learning system as the
  final stage. The use of convolution greatly reduces the
  dimensionality of the problem for image processing, since the
  neural nets at each convolution layer only handle a small patch
  of the input surface.

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

#include "deepconvnet.h"

/**
 * @brief Initialise a deep convnet
 * @param no_of_convolutions Number of layers in the convolution
 *        preprocessing stage
 * @param no_of_deep_layers Number of layers in the deep learner
 * @param inputs_across Horizontal resolution of input images
 * @param inputs_down Vertical resolution of input images
 * @param inputs_depth Depth of input images in bytes
 * @param max_features The number of features learned at each convolution layer
 * @param reduction_factor Reduction factor for convolution layers
 * @param no_of_outputs Number of outputs of the deep learner
 * @param convnet Deep convnet object
 * @param error_threshold Array containing learning thresholds (percent error)
 *        for each layer of both the convolution stage and the deep learner
 * @param random_seed Random number generator seed
 * @return zero on success
 */
int deepconvnet_init(int no_of_convolutions,
                     int no_of_deep_layers,
                     int inputs_across,
                     int inputs_down,
                     int inputs_depth,
                     int max_features,
                     int reduction_factor,
                     int no_of_outputs,
                     deepconvnet * convnet,
                     float error_threshold[],
                     unsigned int * random_seed)
{
    convnet->convolution = (deeplearn_conv*)malloc(sizeof(deeplearn_conv));
    if (!convnet->convolution) return -1;
    if (conv_init(no_of_convolutions,
                  inputs_across,
                  inputs_down,
                  inputs_depth,
                  max_features,
                  reduction_factor, 2,
                  convnet->convolution,
                  error_threshold,
                  random_seed) != 0) {
        return -2;
    }

    convnet->learner = (deeplearn*)malloc(sizeof(deeplearn));
    if (!convnet->learner) return -3;
    if (deeplearn_init(convnet->learner,
                       conv_outputs(convnet->convolution),
                       conv_outputs(convnet->convolution)*8/10,
                       no_of_deep_layers,
                       no_of_outputs,
                       &error_threshold[no_of_convolutions],
                       random_seed) != 0) {
        return -4;
    }

    convnet->training_complete = 0;
    convnet->no_of_images = 0;
    convnet->images = NULL;
    convnet->classifications = NULL;
    convnet->classification_number = NULL;

    /* clear history */
    convnet->history_index = 0;
    convnet->history_ctr = 0;
    convnet->history_step = 1;

    /* default history settings */
    convnet->BPerror = DEEPLEARN_UNKNOWN_ERROR;
    convnet->training_ctr = 0;
    convnet->history_plot_interval = 30;
    sprintf(convnet->history_plot_filename,"%s","training.png");
    sprintf(convnet->history_plot_title,"%s","Training History");

    convnet->current_layer = 0;
    convnet->training_set_index = NULL;
    convnet->test_set_index = NULL;
    return 0;
}

/**
 * @brief Is the given convnet still training?
 * @param convnet Deep convnet object
 * @returns True if still training
 */
int deepconvnet_is_training(deepconvnet * convnet)
{
    return ((convnet->convolution->training_complete != 0) ||
            (convnet->learner->training_complete != 0));
}

/**
 * @brief Frees memory
 * @param convnet Deep convnet object
 */
void deepconvnet_free(deepconvnet * convnet)
{
    conv_free(convnet->convolution);
    free(convnet->convolution);

    deeplearn_free(convnet->learner);
    free(convnet->learner);

    if (convnet->no_of_images > 0) {
        for (int i = 0; i < convnet->no_of_images; i++) {
            if (convnet->images[i] != NULL) {
                free(convnet->images[i]);
                convnet->images[i] = 0;
            }
            free(convnet->classifications[i]);
        }
        free(convnet->images);
        free(convnet->classifications);
        free(convnet->classification_number);
        convnet->no_of_images = 0;
    }

    if (convnet->training_set_index != NULL) {
        free(convnet->training_set_index);
    }
    if (convnet->test_set_index != NULL) {
        free(convnet->test_set_index);
    }
}

/**
 * @brief Update the learning history
 * @param convnet Deep convnet object
 */
static void deepconvnet_update_history(deepconvnet * convnet)
{
    int i;
    float error_value;

    if (convnet->history_step == 0) return;

    convnet->history_ctr++;
    if (convnet->history_ctr >= convnet->history_step) {
        if (convnet->convolution->training_complete == 0) {
            error_value = convnet->convolution->BPerror;
        }
        else {
            error_value = convnet->learner->BPerror;
        }
        if (error_value == DEEPLEARN_UNKNOWN_ERROR) {
            error_value = 0;
        }

        convnet->history[convnet->history_index] = error_value;
        convnet->history_index++;
        convnet->history_ctr = 0;
        convnet->BPerror = error_value;

        if (convnet->history_index >= DEEPLEARN_HISTORY_SIZE) {
            for (i = 0; i < convnet->history_index; i++) {
                convnet->history[i/2] = convnet->history[i];
            }
            convnet->history_index /= 2;
            convnet->history_step *= 2;
        }
    }
}

/**
 * @brief Saves the given deep convnet object to a file
 * @param fp File pointer
 * @param convnet Deep convnet object
 * @return zero value on success
 */
int deepconvnet_save(FILE * fp, deepconvnet * convnet)
{
    if (conv_save(fp, convnet->convolution) != 0) return -1;
    if (deeplearn_save(fp, convnet->learner) != 0) return -2;
    return 0;
}

/**
 * @brief Loads a deep convnet object from file
 * @param fp File pointer
 * @param convnet Deep convnet object
 * @param random_seed Random number generator seed
 * @return zero value on success
 */
int deepconvnet_load(FILE * fp, deepconvnet * convnet,
                     unsigned int * random_seed)
{
    if (conv_load(fp, convnet->convolution) != 0) return -1;
    if (deeplearn_load(fp, convnet->learner, random_seed) != 0) {
        return -2;
    }
    return 0;
}

/**
 * @brief Sets the input values to the outputs of convolutional net.
 *        This provides the glue sticking the preprocessing convolutional
 *        stage together with the final deep learning stage.
 * @param learner Deep learner object
 * @param conv Convolution object
 * @return zero on success
 */
static int deepconvnet_set_inputs_conv(deeplearn * learner, deeplearn_conv * conv)
{
    int layer_index = conv->no_of_layers-1;
    int width = conv_layer_width(layer_index, conv, 1);
    int height = conv_layer_height(layer_index, conv, 1);

    if (learner->net->NoOfInputs !=
        width * height * conv->max_features) {
        return -1;
    }
    /* NOTE: can this be done with memcpy ? */
    for (int i = 0; i < learner->net->NoOfInputs; i++) {
        deeplearn_set_input(learner, i, conv->layer[layer_index].pooling[i]);
    }

    return 0;
}

/**
 * @brief Updates the current training error
 * @param convnet Deep convnet object
 */
static void deepconvnet_update_training_error(deepconvnet * convnet)
{
    /* update the backprop error */
    if (convnet->current_layer <
        convnet->convolution->no_of_layers) {
        convnet->BPerror = convnet->convolution->BPerror;
    }
    else {
        convnet->BPerror = convnet->learner->BPerror;
    }
}

/**
 * @brief Updates the index of the current layer being trained
 * @param convnet Deep convnet object
 */
static void deepconvnet_update_current_layer(deepconvnet * convnet)
{
    /* set the current layer */
    convnet->current_layer =
        convnet->convolution->current_layer +
        convnet->learner->current_hidden_layer;
}

/**
 * @brief Periodicaly plots the training error in a graph
 * @param convnet Deep convnet object
 */
static void deepconvnet_update_training_plot(deepconvnet * convnet)
{
    /* plot a graph showing training progress */
    if (convnet->history_plot_interval > 0) {
        if (convnet->training_ctr > convnet->history_plot_interval) {
            if (strlen(convnet->history_plot_filename) > 0) {
                deepconvnet_plot_history(convnet, 1024, 480);
            }
            convnet->training_ctr = 0;
        }
        convnet->training_ctr++;
    }
}

/**
 * @brief Used during training to update error, current layer, etc
 * @param convnet Deep convnet object
 */
static void deepconvnet_update(deepconvnet * convnet)
{
    deepconvnet_update_training_error(convnet);
    deepconvnet_update_current_layer(convnet);
    deepconvnet_update_history(convnet);
    deepconvnet_update_training_plot(convnet);
    convnet->training_complete = convnet->learner->training_complete;
}

/**
 * @brief Update routine for training the system
 * @param convnet Deep convnet object
 * @param img Array containing input image
 * @param class_number Desired class number
 * @return Zero on success
 */
int deepconvnet_update_img(deepconvnet * convnet, unsigned char img[],
                           int class_number)
{
    unsigned char use_dropouts = 0;

    if (deepconvnet_is_training(convnet)) use_dropouts = 1;

    if (conv_img(img, convnet->convolution, use_dropouts) != 0) {
        return -1;
    }

    if (convnet->convolution->training_complete == 0) {
        deepconvnet_update(convnet);
        return 0;
    }

    if (deepconvnet_set_inputs_conv(convnet->learner,
                                    convnet->convolution) != 0) {
        return -2;
    }

    if (convnet->learner->training_complete == 0) {
        if (deeplearn_training_last_layer(convnet->learner)) {
            deepconvnet_set_class(convnet, class_number);
        }
        deeplearn_update(convnet->learner);
        deepconvnet_update(convnet);
    }
    else {
        /* feed forward only */
        deeplearn_feed_forward(convnet->learner);
    }
    return 0;
}

/**
 * @brief Test a convnet with an unknown image
 * @param convnet Deep convnet object
 * @param img Array containing input image
 * @return Zero on success
 */
int deepconvnet_test_img(deepconvnet * convnet, unsigned char img[])
{
    const unsigned char use_dropouts = 0;

	if (conv_img(img, convnet->convolution, use_dropouts) != 0) {
		return -1;
	}

	if (deepconvnet_set_inputs_conv(convnet->learner,
									convnet->convolution) != 0) {
		return -2;
	}

	/* feed forward only */
	deeplearn_feed_forward(convnet->learner);
	return 0;
}

/**
 * @brief Sets a desired output value
 * @param convnet Deep convnet object
 * @param index Index of the output unit
 * @param value Value to set in the range 0.0 - 1.0
 */
void deepconvnet_set_output(deepconvnet * convnet, int index, float value)
{
	deeplearn_set_output(convnet->learner, index, value);
}

/**
 * @brief Sets a desired output class
 * @param convnet Deep convnet object
 * @param class_number The class number (output index to be set active)
 */
void deepconvnet_set_class(deepconvnet * convnet, int class_number)
{
	deeplearn_set_class(convnet->learner, class_number);
}

/**
 * @brief Returns the value of an output
 * @param convnet Deep convnet object
 * @param index Index of the output unit
 * @return Value of the output
 */
float deepconvnet_get_output(deepconvnet * convnet, int index)
{
	return deeplearn_get_output(convnet->learner, index);
}

/**
 * @brief Returns the output class
 * @param convnet Deep convnet object
 * @return The class number (output index to be set active)
 */
int deepconvnet_get_class(deepconvnet * convnet)
{
	return deeplearn_get_class(convnet->learner);
}

/**
 * @brief Sets the learning rate
 * @param convnet Deep convnet object
 * @param rate the learning rate in the range 0.0 to 1.0
 */
void deepconvnet_set_learning_rate(deepconvnet * convnet, float rate)
{
	conv_set_learning_rate(convnet->convolution, rate);
	deeplearn_set_learning_rate(convnet->learner, rate);
}

/**
 * @brief Sets the percentage of units which drop out during training
 * @param convnet Deep convnet object
 * @param dropout_percent Percentage of units which drop out in the range 0 to 100
 */
void deepconvnet_set_dropouts(deepconvnet * convnet, float dropout_percent)
{
	conv_set_dropouts(convnet->convolution, dropout_percent);
	deeplearn_set_dropouts(convnet->learner, dropout_percent);
}

/**
 * @brief Uses gnuplot to plot the training error for the given learner
 * @param convnet Deep convnet object
 * @param filename Filename for the image to save as
 * @param title Title of the graph
 * @param image_width Width of the image in pixels
 * @param image_height Height of the image in pixels
 * @return zero on success
 */
int deepconvnet_plot_history(deepconvnet * convnet,
							 int image_width, int image_height)
{
	int index,retval=0;
	FILE * fp;
	char data_filename[256];
	char plot_filename[256];
	char command_str[256];
	float value;
	float max_value = 0.01f;

	sprintf(data_filename,"%s%s",DEEPLEARN_TEMP_DIRECTORY,
			"libdeep_data.dat");
	sprintf(plot_filename,"%s%s",DEEPLEARN_TEMP_DIRECTORY,
			"libdeep_data.plot");

	/* save the data */
	fp = fopen(data_filename,"w");
	if (!fp) return -1;
	for (index = 0; index < convnet->history_index; index++) {
		value = convnet->history[index];
		fprintf(fp,"%d    %.10f\n",
				index*convnet->history_step,value);
		/* record the maximum error value */
		if (value > max_value) {
			max_value = value;
		}
	}
	fclose(fp);

	/* create a plot file */
	fp = fopen(plot_filename,"w");
	if (!fp) return -1;
	fprintf(fp,"%s","reset\n");
	fprintf(fp,"set title \"%s\"\n",convnet->history_plot_title);
	fprintf(fp,"set xrange [0:%d]\n",
			convnet->history_index * convnet->history_step);
	fprintf(fp,"set yrange [0:%f]\n", max_value*102/100);
	fprintf(fp,"%s","set lmargin 9\n");
	fprintf(fp,"%s","set rmargin 2\n");
	fprintf(fp,"%s","set xlabel \"Time Step\"\n");
	fprintf(fp,"%s","set ylabel \"Training Error Percent\"\n");

	fprintf(fp,"%s","set grid\n");
	fprintf(fp,"%s","set key right top\n");

	fprintf(fp,"set terminal png size %d,%d\n",
			image_width, image_height);
	fprintf(fp,"set output \"%s\"\n", convnet->history_plot_filename);
	fprintf(fp,"plot \"%s\" using 1:2 notitle with lines\n",
			data_filename);
	fclose(fp);

	/* run gnuplot using the created files */
	sprintf(command_str,"gnuplot %s", plot_filename);
	retval = system(command_str); /* I assume this is synchronous */

	/* remove temporary files */
	sprintf(command_str,"rm %s %s", data_filename, plot_filename);
	retval = system(command_str);

	return retval;
}

/**
 * @brief Performs training
 * @param convnet Deep convnet object
 */
int deepconvnet_training(deepconvnet * convnet)
{
	if (convnet->learner->training_complete != 0) return 0;
	if (convnet->no_of_images == 0) return 0;
	if (convnet->classification_number == NULL) return -1;

	/* pick an image at random */
	int training_images = convnet->no_of_images*8/10;
	int index0 =
		rand_num(&convnet->learner->net->random_seed)%training_images;
	int index = convnet->training_set_index[index0];
	unsigned char * img = convnet->images[index];
	if (deepconvnet_update_img(convnet, img,
							   convnet->classification_number[index]) != 0) {
		return -2;
	}

	return 0;
}

/**
 * @brief Returns performance on the test set
 * @param convnet Deep convnet object
 * @return Percentage of correct classifications or a negative number on error
 */
float deepconvnet_get_performance(deepconvnet * convnet)
{
	float performance = 0;
	int ctr = 0;
	int test_images = convnet->no_of_images*2/10;

	if (convnet->no_of_images == 0) return -1;
	if (convnet->classification_number == NULL) return -2;

	for (int i = 0; i < test_images; i++) {
		int index = convnet->test_set_index[i];
		unsigned char * img = convnet->images[index];
		deepconvnet_update_img(convnet, img, -1);
		if (deeplearn_get_class(convnet->learner) ==
			convnet->classification_number[index]) {
			performance += 100.0f;
		}
		ctr++;
	}
	if (ctr > 0) {
		performance /= ctr;
	}
	return performance;
}

/**
 * @brief Creates training and test arrays which contain randomly ordered
 *        indexes to the main images array. This tries to ensure that there
 *        is no bias depending on the sequences of inputs during training.
 * @param convnet Deep convnet object
 */
int deepconvnet_create_training_test_sets(deepconvnet * convnet)
{
	int i, j;

	/* create arrays to store randomly ordered array indexes
	   for training and test sets */
	int training_images = convnet->no_of_images*8/10;
	convnet->training_set_index =
		(int*)malloc((training_images+1)*sizeof(int));
	if (!convnet->training_set_index) return -1;
	i = 0;
	while (i < training_images) {
		int index =
			rand_num(&convnet->learner->net->random_seed)%convnet->no_of_images;
		for (j = 0; j < i; j++) {
			if (convnet->training_set_index[j] == index) {
				break;
			}
		}
		if (j == i) {
			convnet->training_set_index[i] = index;
			i++;
		}
	}
	int test_images = 0;
	convnet->test_set_index =
		(int*)malloc((convnet->no_of_images - training_images+1)*sizeof(int));
	if (!convnet->test_set_index) return -2;
	for (i = 0; i < convnet->no_of_images; i++) {
		for (j = 0; j < training_images; j++) {
			if (convnet->training_set_index[j] == i) {
				break;
			}
		}
		if (j == training_images) {
			convnet->test_set_index[test_images++] = i;
		}
	}
	return 0;
}


/**
 * @brief Reads images from a given directory and creates a deep convnet
 * @param directory Directory containing png images
 * @param convnet Deep convnet object
 * @param image_width Image width
 * @param image_height Image height
 * @param no_of_convolutions The number of convolution layers
 * @param max_features_per_convolution Number of features learned at
 *        each convolution layer
 * @param no_of_deep_layers Number of layers for the deep learner
 * @param no_of_outputs Number of output units
 * @param output_classes The number of output classes if the output in the
 *        data set is a single integer value
 * @param error_threshold Training error thresholds for each hidden layer
 * @param random_seed Random number seed
 * @return zero on success
 */
int deepconvnet_read_images(char * directory,
							deepconvnet * convnet,
							int image_width, int image_height,
							int no_of_convolutions,
							int max_features_per_convolution,
							int reduction_factor,
							int no_of_deep_layers,
							int no_of_outputs,
							int output_classes,
							float error_threshold[],
							unsigned int * random_seed)
{
	if (deepconvnet_init(no_of_convolutions,
						 no_of_deep_layers,
						 image_width, image_height, 3,
						 max_features_per_convolution,
						 reduction_factor,
						 no_of_outputs, convnet,
						 error_threshold,
						 random_seed) != 0) {
		return -1;
	}

	convnet->no_of_images =
		deeplearn_load_training_images(directory, &convnet->images,
									   &convnet->classifications,
									   &convnet->classification_number,
									   image_width, image_height);
	if (convnet->no_of_images <= 0) {
		return -2;
	}

	if (deepconvnet_create_training_test_sets(convnet) != 0) {
		return -3;
	}

	return 0;
}

/**
 * @brief Plots the features learned for a given convolution layer
 * @param convnet Deep convnet object
 * @param layer_index Index number of the convolution layer
 * @param filename Filename to save the image as (must be PNG format)
 * @param img_width Image width
 * @param img_height Image height
 * @return zero on success
 */
int deepconvnet_plot_features(deepconvnet * convnet,
							  int layer_index,
							  char * filename,
							  int img_width, int img_height)
{
	unsigned char * img;

	/* allocate memory for the image */
	img = (unsigned char*)malloc(img_width*img_height*3*sizeof(unsigned char));
	if (!img) {
		return -1;
	}

	int retval =
		conv_plot_features(convnet->convolution,
						   layer_index, img, img_width, img_height);
	if (retval != 0) return retval;

	deeplearn_write_png_file(filename,
							 (unsigned int)img_width,
							 (unsigned int)img_height,
							 24, img);

	free(img);
	return 0;
}
