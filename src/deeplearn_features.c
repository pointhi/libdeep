/*
  libdeep - a library for deep learning
  Copyright (C) 2013-2016  Bob Mottram <bob@robotics.uk.to>

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

#include "deeplearn_features.h"

/**
 * @brief Scans an image patch and transfers the values to an autocoder
 * @param img image array
 * @param img_width Width of the image
 * @param img_depth Depth of the image, typically bytes per pixel
 * @param tx Top left coordinate of the patch
 * @param ty Top coordinate of the patch
 * @param bx Bottom right coordinate of the patch
 * @param by Bottom coordinate of the patch
 * @param feature_autocoder Autocoder object
 * @return zero on success
 */
static int scan_img_patch(unsigned char img[],
                          int img_width, int img_depth,
                          int tx, int ty, int bx, int by,
                          ac * feature_autocoder)
{
    int index_feature_input = 0;
    for (int y = ty; y < by; y++) {
        for (int x = tx; x < bx; x++) {
            int index_img =
                ((y*img_width) + x) * img_depth;
            for (int d = 0; d < img_depth; d++) {
                /* convert from 8 bit to a neuron value */
                autocoder_set_input(feature_autocoder,
                                    index_feature_input++,
                                    PIXEL_TO_FLOAT(img[index_img+d]));
            }
        }
    }

    /* check that the patch size is the same as the autocoder inputs */
    if (index_feature_input != feature_autocoder->NoOfInputs) {
        return -1;
    }
    return 0;
}

/**
 * @brief Create an image patch from autocoder hidden units.
 *        Note that the image array should initially be cleared
 * @param img image array
 * @param img_width Width of the image
 * @param img_depth Depth of the image, typically bytes per pixel
 * @param tx Top left coordinate of the patch
 * @param ty Top coordinate of the patch
 * @param bx Bottom right coordinate of the patch
 * @param by Bottom coordinate of the patch
 * @param feature_autocoder Autocoder object
 * @return zero on success
 */
static int create_img_patch(float img[],
                            int img_width, int img_depth,
                            int tx, int ty, int bx, int by,
                            ac * feature_autocoder)
{
    int i;
    float f;
    
    for (i = 0; i < feature_autocoder->NoOfHiddens; i++) {
        f = autocoder_get_hidden(feature_autocoder, i);
        for (int y = ty; y < by; y++) {
            for (int x = tx; x < bx; x++) {
                int index_img =
                    ((y*img_width) + x) * img_depth;
                for (int d = 0; d < img_depth; d++) {
                    img[index_img+d] +=
                        f * feature_autocoder->weights[index_img+d];                    
                }
            }
        }       
    }
    return 0;
}

/**
 * @brief Scans a patch within a 2D array of floats and transfers the values
 *        to an autocoder
 * @param inputs_floats inputs array
 * @param inputs_width Width of the floats array
 * @param inputs_depth Depth of the floats array
 * @param tx Top left coordinate of the patch
 * @param ty Top coordinate of the patch
 * @param bx Bottom right coordinate of the patch
 * @param by Bottom coordinate of the patch
 * @param feature_autocoder Autocoder object
 * @return zero on success
 */
static int scan_floats_patch(float inputs_floats[],
                             int inputs_width, int inputs_depth,
                             int tx, int ty, int bx, int by,
                             ac * feature_autocoder)
{
    int index_feature_input = 0;
    for (int y = ty; y < by; y++) {
        for (int x = tx; x < bx; x++) {
            int index_inputs =
                ((y*inputs_width) + x) * inputs_depth;

            /* depth typically corresponds to colour channels
               in the initial layer, or feature responses in
               subsequent layers */
            for (int d = 0; d < inputs_depth; d++) {
                /* set the inputs of the autocoder */
                autocoder_set_input(feature_autocoder,
                                    index_feature_input++,
                                    inputs_floats[index_inputs+d]);
            }
        }
    }

/* check that the patch size is the same as the autocoder inputs */
    if (index_feature_input != feature_autocoder->NoOfInputs) {
        return -1;
    }
    return 0;
}

/**
 * @brief Returns the input patch bounding box for an x,y coordinate
 *        within the second layer
 * @param x Position across within the second layer
 * @param y Position down within the second layer
 * @param samples_across The number of units across in the second layer
 * @param samples_down The number of units down in the second layer
 * @param patch_radius The radius of the patch within the input layer
 * @param width Width of the input layer
 * @param height Height of the input layer
 * @param tx Returned top left coordinate
 * @param ty Returned top coordinate
 * @param bx Returned bottom right coordinate
 * @param by Returned bottom coordinate
 * @return zero if the patch does not exceed the limits of the area
 */
int features_patch_coords(int x, int y,
                          int samples_across,
                          int samples_down,
                          int patch_radius,
                          int width, int height,
                          int * tx, int * ty, int * bx, int * by)
{
    int cy = y * height / samples_down;
    int cx = x * width / samples_across;

    *ty = cy - patch_radius;
    *by = cy + patch_radius;
    if (*ty < 0) return -1;
    if (*by >= height) return -2;

    *tx = cx - patch_radius;
    *bx = cx + patch_radius;
    if (*tx < 0) return -3;
    if (*bx >= width) return -4;
    return 0;
}

/**
 * @brief Learn a feature set between an input image and a neuron layer
 * @param samples_across The number of units across in the second layer
 * @param samples_down The number of units down in the second layer
 * @param patch_radius The radius of the patch within the image
 * @param img_width Width of the image
 * @param img_height Height of the image
 * @param img_depth Depth of the image (mono=1, RGB=3)
 * @param img Image buffer
 * @param layer0_units Number of units in the neuron layer
 * @param feature_autocoder An autocoder used for feature learning
 * @param BPerror Returned total learning error
 * @returns zero on success
 */
int features_learn_from_img(int samples_across,
                            int samples_down,
                            int patch_radius,
                            int img_width,
                            int img_height,
                            int img_depth,
                            unsigned char img[],
                            int layer0_units,
                            ac * feature_autocoder,
                            float * BPerror)
{
    int no_of_learned_features = feature_autocoder->NoOfHiddens;
    *BPerror = 0;

    if (samples_across * samples_down * no_of_learned_features !=
        layer0_units) {
        /* across*down doesn't equal the second layer units */
        return -1;
    }

    if (feature_autocoder->NoOfInputs !=
        patch_radius*patch_radius*4*img_depth) {
        /* the patch size doesn't match the feature
           learner inputs */
        printf("NoOfInputs %d\n",feature_autocoder->NoOfInputs);
        printf("patch_radius %d\n",patch_radius);
        printf("img_depth %d\n",img_depth);
        return -2;
    }

    /* for each patch */
    for (int fy = 0; fy < samples_down; fy++) {
        for (int fx = 0; fx < samples_across; fx++) {

            /* get the coordinates of the patch in the image */
            int tx=0, ty=0, bx=0, by=0;
            if (features_patch_coords(fx, fy,
                                      samples_across, samples_down,
                                      patch_radius,
                                      img_width, img_height,
                                      &tx, &ty, &bx, &by) != 0) {
                continue;
            }

            /* scan the patch into the feature autocoder inputs */
            if (scan_img_patch(img, img_width, img_depth,
                               tx, ty, bx, by,
                               feature_autocoder) != 0) {
                return -4;
            }

            /* feature autocoder learns and the total error is incremented */
            autocoder_update(feature_autocoder);
            *BPerror = *BPerror + feature_autocoder->BPerror;
        }
    }

    /* calculate the average error */
    *BPerror = *BPerror / (samples_across*samples_down);
    return 0;
}

/**
 * @brief Learn a feature set between an array of floats and a neuron layer
 *        Inputs are expected to have values in the range 0.25->0.75
 * @param samples_across The number of units across in the second layer
 * @param samples_down The number of units down in the second layer
 * @param patch_radius The radius of the patch within the inputs
 * @param inputs_width Width of the inputs array
 * @param inputs_height Height of the inputs array
 * @param inputs_depth Depth of the inputs array
 * @param inputs_floats Inputs buffer of floats
 * @param layer0_units Number of units in the neuron layer
 * @param feature_autocoder An autocoder used for feature learning
 * @param BPerror Returned total backprop error
 * @returns zero on success
 */
int features_learn_from_flt(int samples_across,
                            int samples_down,
                            int patch_radius,
                            int inputs_width,
                            int inputs_height,
                            int inputs_depth,
                            float inputs_floats[],
                            int layer0_units,
                            ac * feature_autocoder,
                            float * BPerror)
{
    int no_of_learned_features = feature_autocoder->NoOfHiddens;
    *BPerror = 0;

    if (samples_across * samples_down * no_of_learned_features !=
        layer0_units) {
        /* across*down doesn't equal the second layer units */
        return -1;
    }

    if (feature_autocoder->NoOfInputs !=
        patch_radius*patch_radius*4*inputs_depth) {
        /* the patch size doesn't match the feature
           learner inputs */
        return -2;
    }

    /* for each patch */
    for (int fy = 0; fy < samples_down; fy++) {
        for (int fx = 0; fx < samples_across; fx++) {

            /* get the coordinates of the patch in the image */
            int tx=0, ty=0, bx=0, by=0;
            if (features_patch_coords(fx, fy, samples_across,
                                      samples_down,
                                      patch_radius,
                                      inputs_width, inputs_height,
                                      &tx, &ty, &bx, &by) != 0) {
                continue;
            }

            /* scan the patch into the feature autocoder inputs */
            if (scan_floats_patch(inputs_floats,
                                  inputs_width, inputs_depth,
                                  tx, ty, bx, by,
                                  feature_autocoder) != 0) {
                return -4;
            }

            /* feature autocoder learns and the total error is incremented */
            autocoder_update(feature_autocoder);
            *BPerror = *BPerror + feature_autocoder->BPerror;
        }
    }

    /* calculate the average error */
    *BPerror = *BPerror / (samples_across*samples_down);
    return 0;
}

/**
 * @brief Convolve an image with learned features and output
 *        the results to the input layer of a neural net
 * @param samples_across The number of units across in the input layer
 *        (sampling grid resolution)
 * @param samples_down The number of units down in the input layer
 *        (sampling grid resolution)
 * @param patch_radius The radius of the patch within the image
 * @param img_width Width of the image
 * @param img_height Height of the image
 * @param img_depth Depth of the image (mono=1, RGB=3)
 * @param img Image buffer
 * @param layer0 Neural net
 * @param feature_autocoder An autocoder containing learned features
 * @param use_dropouts non-zero if dropouts are used
 * @returns zero on success
 */
int features_conv_img_to_neurons(int samples_across,
                                 int samples_down,
                                 int patch_radius,
                                 int img_width,
                                 int img_height,
                                 int img_depth,
                                 unsigned char img[],
                                 bp * net,
                                 ac * feature_autocoder,
                                 unsigned char use_dropouts)
{
    int no_of_learned_features = feature_autocoder->NoOfHiddens;

    if (samples_across * samples_down * no_of_learned_features !=
        net->NoOfInputs) {
        /* across*down doesn't equal the second layer units */
        return -1;
    }

    if (feature_autocoder->NoOfInputs !=
        patch_radius*patch_radius*4*img_depth) {
        /* the patch size doesn't match the feature
           learner inputs */
        return -2;
    }

    for (int fy = 0; fy < samples_down; fy++) {
        for (int fx = 0; fx < samples_across; fx++) {
            int tx=0, ty=0, bx=0, by=0;
            if (features_patch_coords(fx, fy,
                                      samples_across, samples_down,
                                      patch_radius,
                                      img_width, img_height,
                                      &tx, &ty, &bx, &by) != 0) {
                continue;
            }

            if (scan_img_patch(img,
                               img_width, img_depth,
                               tx, ty, bx, by,
                               feature_autocoder) != 0) {
                return -4;
            }

            int index_input_layer =
                (fy * samples_across + fx) *
                no_of_learned_features;
            autocoder_encode(feature_autocoder, feature_autocoder->hiddens,
                             use_dropouts);
            for (int f = 0; f < no_of_learned_features; f++) {
                bp_set_input(net, index_input_layer+f,
                             autocoder_get_hidden(feature_autocoder, f));
            }
        }
    }
    return 0;
}

/**
 * @brief Convolve an image with learned features and output
 *        the results to an array of floats
 * @param samples_across The number of units across in the array of floats
 *        (sampling grid resolution)
 * @param samples_down The number of units down in the array of floats
 *        (sampling grid resolution)
 * @param patch_radius The radius of the patch within the float array
 * @param img_width Width of the image
 * @param img_height Height of the image
 * @param img_depth Depth of the image (mono=1, RGB=3)
 * @param img Image buffer
 * @param layer0_units Number of units in the float array
 * @param layer0 float array
 * @param feature_autocoder An autocoder containing learned features
 * @param use_dropouts Non-zero if dropouts are to be used
 * @returns zero on success
 */
int features_conv_img_to_flt(int samples_across,
                             int samples_down,
                             int patch_radius,
                             int img_width,
                             int img_height,
                             int img_depth,
                             unsigned char img[],
                             int layer0_units,
                             float layer0[],
                             ac * feature_autocoder,
                             unsigned char use_dropouts)
{
    int no_of_learned_features = feature_autocoder->NoOfHiddens;

    if (samples_across * samples_down * no_of_learned_features !=
        layer0_units) {
        /* across*down doesn't equal the second layer units */
        return -1;
    }

    if (feature_autocoder->NoOfInputs !=
        patch_radius*patch_radius*4*img_depth) {
        /* the patch size doesn't match the feature
           learner inputs */
        return -2;
    }

    /* for each input image sample */
    for (int fy = 0; fy < samples_down; fy++) {
        for (int fx = 0; fx < samples_across; fx++) {
            /* starting position in the first layer,
               where the depth is the number of encoded features */
            int index_layer0 =
                (fy * samples_across + fx) *
                no_of_learned_features;

            /* coordinates of the patch in the input image */
            int tx=0, ty=0, bx=0, by=0;
            if (features_patch_coords(fx, fy,
                                      samples_across, samples_down,
                                      patch_radius,
                                      img_width, img_height,
                                      &tx, &ty, &bx, &by) != 0) {
                for (int f = 0; f < no_of_learned_features; f++) {
                    layer0[index_layer0+f] = 0;
                }
                continue;
            }

            /* scan the patch from the input image and get the
               feature responses */
            if (scan_img_patch(img, img_width, img_depth,
                               tx, ty, bx, by,
                               feature_autocoder) != 0) {
                return -4;
            }

            /* set the first layer at this position to the feature responses */
            autocoder_encode(feature_autocoder, &layer0[index_layer0],
                             use_dropouts);
        }
    }
    return 0;
}

/**
 * @brief Deconvolve a float image with learned features and output
 *        the results to an array of floats
 * @param samples_across The number of units across in the array of floats
 *        (sampling grid resolution)
 * @param samples_down The number of units down in the array of floats
 *        (sampling grid resolution)
 * @param patch_radius The radius of the patch within the float array
 * @param img_width Width of the image
 * @param img_height Height of the image
 * @param img_depth Depth of the image (mono=1, RGB=3)
 * @param img Image buffer
 * @param layer_units Number of units in the layer array
 * @param layer float array
 * @param feature_autocoder An autocoder containing learned features
 * @returns zero on success
 */
int features_deconv_flt_to_flt(int samples_across,
							   int samples_down,
							   int patch_radius,
							   int img_width,
							   int img_height,
							   int img_depth,
							   float img[],
							   int layer_units,
							   float layer[],
							   ac * feature_autocoder)
{	
    int no_of_learned_features = feature_autocoder->NoOfHiddens;

    
    if (feature_autocoder->NoOfInputs !=
        patch_radius*patch_radius*4*img_depth) {
        /* the patch size doesn't match the feature
           learner inputs */
        return -2;
    }

    /* clear the original image */
    memset((void*)img, '\0', img_width*img_height*img_depth*sizeof(float));
    
    /* for each input image sample */
    for (int fy = 0; fy < samples_down; fy++) {
        for (int fx = 0; fx < samples_across; fx++) {   
            /* starting position in the first layer,
               where the depth is the number of encoded features */
            int index_layer =
                (fy * samples_across + fx) *
                no_of_learned_features;

            /* coordinates of the patch in the input image */
            int tx=0, ty=0, bx=0, by=0;
            if (features_patch_coords(fx, fy,
                                      samples_across, samples_down,
                                      patch_radius,
                                      img_width, img_height,
                                      &tx, &ty, &bx, &by) != 0) {
                /* set the hidden unit values from the previous player */
                for (int f = 0; f < no_of_learned_features; f++) {
                    autocoder_set_hidden(feature_autocoder, f,
                                         layer[index_layer+f]);
                }
                continue;
            }

            /* create the patch from the autocoder hidden units */
            if (create_img_patch(img, img_width, img_depth,
                                 tx, ty, bx, by,
                                 feature_autocoder) != 0) {
                return -4;
            }
        }
    }
    
    return 0;
}


/**
 * @brief Deconvolve an image with learned features and output
 *        the results to an array of floats
 * @param samples_across The number of units across in the array of floats
 *        (sampling grid resolution)
 * @param samples_down The number of units down in the array of floats
 *        (sampling grid resolution)
 * @param patch_radius The radius of the patch within the float array
 * @param img_width Width of the image
 * @param img_height Height of the image
 * @param img_depth Depth of the image (mono=1, RGB=3)
 * @param img Image buffer
 * @param layer_units Number of units in the layer array
 * @param layer float array
 * @param feature_autocoder An autocoder containing learned features
 * @returns zero on success
 */
int features_deconv_img_to_flt(int samples_across,
                               int samples_down,
                               int patch_radius,
                               int img_width,
                               int img_height,
                               int img_depth,
                               unsigned char img[],
                               int layer_units,
                               float layer[],
                               ac * feature_autocoder)
{
    int retval, i;

    
    if (feature_autocoder->NoOfInputs !=
        patch_radius*patch_radius*4*img_depth) {
        /* the patch size doesn't match the feature
           learner inputs */
        return -2;
    }

    /* create a temporary floats image */
    float * deconv_img =
        (float*)malloc(img_width*img_height*img_depth*sizeof(float));
    memset((void*)deconv_img, '\0', img_width*img_height*img_depth*sizeof(float));

    /* clear the original image */
    memset((void*)img, '\0', img_width*img_height*img_depth*sizeof(unsigned char));

    retval =
        features_deconv_flt_to_flt(samples_across,
                                   samples_down,
                                   patch_radius,
                                   img_width,
                                   img_height,
                                   img_depth,
                                   deconv_img,
                                   layer_units, layer,
                                   feature_autocoder);
    if (retval != 0) {
        free(deconv_img);
        return retval;
    }

    for (i = 0; i < img_width*img_height*img_depth; i++) {
        if ((deconv_img[i] > 0) && (deconv_img[i] <= 255)) {
            img[i] = (unsigned char)deconv_img[i];
        }
        else {
            if (deconv_img[i] > 255) {
                img[i] = 255;
            }           
        }
    }
    
    free(deconv_img);
    return 0;
}

/**
 * @brief Convolve a first array of floats to a second one
 * @param samples_across The number of units across in the second array of floats (sampling grid resolution)
 * @param samples_down The number of units down in the second array of floats (sampling grid resolution)
 * @param patch_radius The radius of the patch within the first float array
 * @param floats_width Width of the image
 * @param floats_height Height of the image
 * @param floats_depth Depth of the image (mono=1, RGB=3)
 * @param layer0 First array of floats
 * @param layer1_units Number of units in the second float array
 * @param layer1 Second float array
 * @param feature_autocoder An autocoder containing learned features
 * @param use_dropouts non-zero if dropouts are to be used
 * @returns zero on success
 */
int features_conv_flt_to_flt(int samples_across,
                             int samples_down,
                             int patch_radius,
                             int floats_width,
                             int floats_height,
                             int floats_depth,
                             float layer0[],
                             int layer1_units,
                             float layer1[],
                             ac * feature_autocoder,
                             unsigned char use_dropouts)
{
    int no_of_learned_features = feature_autocoder->NoOfHiddens;

    if (samples_across * samples_down * no_of_learned_features !=
        layer1_units) {
        /* across*down doesn't equal the second layer units */
        return -1;
    }

    if (feature_autocoder->NoOfInputs !=
        patch_radius*patch_radius*4*floats_depth) {
        /* the patch size doesn't match the feature
           learner inputs */
        return -2;
    }

    for (int fy = 0; fy < samples_down; fy++) {
        for (int fx = 0; fx < samples_across; fx++) {
            int index_layer1 =
                (fy * samples_across + fx) *
                no_of_learned_features;
            int tx=0, ty=0, bx=0, by=0;
            if (features_patch_coords(fx, fy,
                                      samples_across, samples_down,
                                      patch_radius,
                                      floats_width, floats_height,
                                      &tx, &ty, &bx, &by) != 0) {
                for (int f = 0; f < no_of_learned_features; f++) {
                    layer1[index_layer1+f] = 0;
                }
                continue;
            }

            if (scan_floats_patch(layer0,
                                  floats_width, floats_depth,
                                  tx, ty, bx, by,
                                  feature_autocoder) != 0) {
                return -4;
            }

            autocoder_encode(feature_autocoder, &layer1[index_layer1],
                             use_dropouts);
        }
    }
    return 0;
}

/**
 * @brief Convolve an array of floats to the input layer of a neural net
 * @param samples_across The number of units across in the layer of neurons
 *        (sampling grid resolution)
 * @param samples_down The number of units down in the layer of neurons
 *        (sampling grid resolution)
 * @param patch_radius The radius of the patch within the float array
 * @param floats_width Width of the image
 * @param floats_height Height of the image
 * @param floats_depth Depth of the image (mono=1, RGB=3)
 * @param layer0 Array of floats
 * @param net Neural net to set the inputs for
 * @param feature_autocoder An autocoder containing learned features
 * @param use_dropouts Non-zero if dropouts are to be used
 * @returns zero on success
 */
int features_conv_floats_to_neurons(int samples_across,
                                    int samples_down,
                                    int patch_radius,
                                    int floats_width,
                                    int floats_height,
                                    int floats_depth,
                                    float layer0[],
                                    bp * net,
                                    ac * feature_autocoder,
                                    unsigned char use_dropouts)
{
    int no_of_learned_features = feature_autocoder->NoOfHiddens;

    if (samples_across * samples_down * no_of_learned_features !=
        net->NoOfInputs) {
        /* across*down doesn't equal the second layer units */
        return -1;
    }

    if (feature_autocoder->NoOfInputs !=
        patch_radius*patch_radius*4*floats_depth) {
        /* the patch size doesn't match the feature
           learner inputs */
        return -2;
    }

    for (int fy = 0; fy < samples_down; fy++) {
        for (int fx = 0; fx < samples_across; fx++) {
            int tx=0, ty=0, bx=0, by=0;
            if (features_patch_coords(fx, fy,
                                      samples_across, samples_down,
                                      patch_radius,
                                      floats_width, floats_height,
                                      &tx, &ty, &bx, &by) != 0) {
                continue;
            }

            if (scan_floats_patch(layer0,
                                  floats_width, floats_depth,
                                  tx, ty, bx, by,
                                  feature_autocoder) != 0) {
                return -4;
            }

            int index_net_inputs =
                (fy * samples_across + fx) *
                no_of_learned_features;
            autocoder_encode(feature_autocoder, feature_autocoder->hiddens,
                             use_dropouts);
            for (int f = 0; f < no_of_learned_features; f++) {
                bp_set_input(net, index_net_inputs+f,
                             autocoder_get_hidden(feature_autocoder, f));
            }
        }
    }
    return 0;
}
