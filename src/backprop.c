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

#include "backprop.h"

/**
* @brief The number of hidden units varies over the hidden layers
*        such that you have an increasing data compression effect
* @param net Backprop neural net object
* @param layer Index number of the hidden layer
* @returns The number of units in the hidden layer
*/
int bp_hiddens_in_layer(bp * net, int layer)
{
    if (layer == 0)
        return net->NoOfHiddens;

    if (layer >= net->HiddenLayers)
        layer = net->HiddenLayers-1;

    return net->NoOfHiddens -
        ((net->NoOfHiddens - net->NoOfOutputs)*layer/net->HiddenLayers);
}

/**
* @brief Initialise a backprop neural net
* @param net Backprop neural net object
* @param no_of_inputs The number of input units
* @param no_of_hiddens The number of units in each hidden layer
* @param hidden_layers The number of hidden layers
* @param no_of_outputs The number of output units
* @param random_seed The random number generator seed
* @returns zero on success
*/
int bp_init(bp * net,
            int no_of_inputs,
            int no_of_hiddens,
            int hidden_layers,
            int no_of_outputs,
            unsigned int * random_seed)
{
    bp_neuron * n;

    net->learningRate = 0.2f;
    net->noise = 0.0f;
    net->random_seed = *random_seed;
    net->BPerror = DEEPLEARN_UNKNOWN_ERROR;
    net->BPerrorAverage = DEEPLEARN_UNKNOWN_ERROR;
    net->BPerrorTotal = DEEPLEARN_UNKNOWN_ERROR;
    net->itterations = 0;
    net->DropoutPercent = 20;

    net->NoOfInputs = no_of_inputs;
    NEURON_ARRAY_ALLOC(net->inputs, no_of_inputs);
    if (!net->inputs)
        return -1;

    net->NoOfHiddens = no_of_hiddens;
    net->NoOfOutputs = no_of_outputs;
    net->HiddenLayers = hidden_layers;
    NEURON_LAYERS_ALLOC(net->hiddens, hidden_layers);
    if (!net->hiddens)
        return -2;

    COUNTDOWN(l, hidden_layers) {
        NEURON_ARRAY_ALLOC(net->hiddens[l], bp_hiddens_in_layer(net,l));
        if (!net->hiddens[l])
            return -3;
    }

    NEURON_ARRAY_ALLOC(net->outputs, no_of_outputs);
    if (!net->outputs)
        return -4;

    /* create inputs */
    COUNTDOWN(i, net->NoOfInputs) {
        NEURONALLOC(net->inputs[i]);
        if (!net->inputs[i])
            return -5;

        if (bp_neuron_init(net->inputs[i], 1, random_seed) != 0)
            return -6;
    }

    /* create hiddens */
    COUNTUP(l, hidden_layers) {
        COUNTUP(i, bp_hiddens_in_layer(net,l)) {
            net->hiddens[l][i] =
                (bp_neuron*)malloc(sizeof(bp_neuron));
            if (!net->hiddens[l][i])
                return -7;

            n = net->hiddens[l][i];
            if (l == 0) {
                if (bp_neuron_init(n, no_of_inputs, random_seed) != 0)
                    return -8;

                /* connect to input layer */
                COUNTDOWN(j, net->NoOfInputs)
                    bp_neuron_add_connection(n, j, net->inputs[j]);
            }
            else {
                if (bp_neuron_init(n, bp_hiddens_in_layer(net,l-1),
                                   random_seed) != 0)
                    return -9;

                /* connect to previous hidden layer */
                COUNTDOWN(j, bp_hiddens_in_layer(net,l-1))
                    bp_neuron_add_connection(n, j, net->hiddens[l-1][j]);
            }
        }
    }

    /* create outputs */
    COUNTDOWN(i, net->NoOfOutputs) {
        NEURONALLOC(net->outputs[i]);
        if (!net->outputs[i])
            return -10;

        n = net->outputs[i];
        if (bp_neuron_init(n, bp_hiddens_in_layer(net,hidden_layers-1),
                           random_seed) != 0)
            return -11;

        COUNTDOWN(j, bp_hiddens_in_layer(net,hidden_layers-1))
            bp_neuron_add_connection(n, j, net->hiddens[hidden_layers-1][j]);
    }
    return 0;
}

/**
* @brief Deallocate the memory for a backprop neural net object
* @param net Backprop neural net object
*/
void bp_free(bp * net)
{
    COUNTDOWN(i, net->NoOfInputs) {
        bp_neuron_free(net->inputs[i]);
        free(net->inputs[i]);
        net->inputs[i] = 0;
    }
    free(net->inputs);

    COUNTDOWN(l, net->HiddenLayers) {
        COUNTDOWN(i, bp_hiddens_in_layer(net,l)) {
            bp_neuron_free(net->hiddens[l][i]);
            free(net->hiddens[l][i]);
            net->hiddens[l][i] = 0;
        }
        free(net->hiddens[l]);
        net->hiddens[l] = 0;
    }
    free(net->hiddens);

    COUNTDOWN(i, net->NoOfOutputs) {
        bp_neuron_free(net->outputs[i]);
        free(net->outputs[i]);
        net->outputs[i] = 0;
    }
    free(net->outputs);
}

/**
* @brief Propagates the current inputs through the layers of the network
* @param net Backprop neural net object
*/
void bp_feed_forward(bp * net)
{
    /* for each hidden layer */
    COUNTUP(l, net->HiddenLayers) {
        /* For each unit within the layer */
        COUNTDOWN(i, bp_hiddens_in_layer(net,l))
            bp_neuron_feedForward(net->hiddens[l][i],
                                  net->noise, &net->random_seed);
    }

    /* for each unit in the output layer */
    COUNTDOWN(i, net->NoOfOutputs)
        bp_neuron_feedForward(net->outputs[i], net->noise, &net->random_seed);
}

/**
* @brief Propagates the current inputs through a given number of
*        layers of the network
* @param net Backprop neural net object
* @param layers The number of layers to propagate through
*/
void bp_feed_forward_layers(bp * net, int layers)
{
    /* for each hidden layer */
    COUNTUP(l, layers) {
        /* if this layer is a hidden layer */
        if (l < net->HiddenLayers) {
            /* For each unit within the layer */
            COUNTDOWN(i, bp_hiddens_in_layer(net,l))
                bp_neuron_feedForward(net->hiddens[l][i],
                                      net->noise, &net->random_seed);
        }
        else {
            /* For each unit within the output layer */
            COUNTDOWN(i, net->NoOfOutputs)
                bp_neuron_feedForward(net->outputs[i],
                                      net->noise, &net->random_seed);
        }
    }
}

/**
* @brief back-propogate errors from the output layer towards the input layer
* @param net Backprop neural net object
*/
void bp_backprop(bp * net, int current_hidden_layer)
{
    int neuron_count=0;
    bp_neuron * n;
    int start_hidden_layer = current_hidden_layer-1;
    float errorPercent=0;

    /* clear all previous backprop errors */
    COUNTDOWN(i, net->NoOfInputs)
        net->inputs[i]->BPerror = 0;

    /* for every hidden layer */
    if (start_hidden_layer < 0)
        start_hidden_layer = 0;

    FOR(l, start_hidden_layer, net->HiddenLayers) {
        /* For each unit within the layer */
        COUNTDOWN(i, bp_hiddens_in_layer(net,l))
            net->hiddens[l][i]->BPerror = 0;
    }

    /* now back-propogate the error from the output units */
    net->BPerrorTotal = 0;
    /* for every output unit */
    COUNTDOWN(i, net->NoOfOutputs) {
        /* get the neuron object */
        n = net->outputs[i];

        /* backpropogate the error */
        bp_neuron_backprop(n);

        /* update the total error which is used to assess
            network performance */
        net->BPerrorTotal += n->BPerror;
        errorPercent += fabs(n->BPerror);
        neuron_count++;
    }

    /* error percentage assuming an encoding range
       of 0.25 -> 0.75 */
    errorPercent = errorPercent * 100 / (0.5f*net->NoOfOutputs);

    /* error on the output units */
    net->BPerror = fabs(net->BPerrorTotal / net->NoOfOutputs);

    /* update the running average */
    if (net->BPerrorAverage == DEEPLEARN_UNKNOWN_ERROR) {
        net->BPerrorAverage = net->BPerror;
        net->BPerrorPercent = errorPercent;
    }
    else {
        net->BPerrorAverage =
            (net->BPerrorAverage*0.999f) +
            (net->BPerror*0.001f);

        net->BPerrorPercent =
            (net->BPerrorPercent*0.999f) +
            (errorPercent*0.001f);
    }

    /* back-propogate through the hidden layers */
    for (int l = net->HiddenLayers-1; l >= start_hidden_layer; l--) {
        /* for every unit in the hidden layer */
        COUNTDOWN(i, bp_hiddens_in_layer(net,l)) {
            /* get the neuron object */
            n = net->hiddens[l][i];

            /* backpropogate the error */
            bp_neuron_backprop(n);

            /* update the total error which is used to assess
                network performance */
            net->BPerrorTotal += n->BPerror;
            neuron_count++;
        }
    }

    /* overall average error */
    net->BPerrorTotal =
        fabs(net->BPerrorTotal / neuron_count);

    /* increment the number of training itterations */
    if (net->itterations < UINT_MAX)
        net->itterations++;
}

/**
 * @brief Reprojects the input layer from a given hidden layer neuron
 *        This is like feedforward in reverse, and allows you
 *        to visualise what a hidden layer neuron is representing
 * @param layer The hidden layer within which the neuron resides
 * @param neuron_index The hidden layer index of the neuron to be reprojected
 */
void bp_reproject(bp * net, int layer, int neuron_index)
{
    bp_neuron * n;

    /* clear all previous backprop errors */
    COUNTDOWN(i, net->NoOfInputs)
        net->inputs[i]->value_reprojected = 0;

    /* for every hidden layer */
    COUNTDOWN(l, layer) {
        /* For each unit within the layer */
        COUNTDOWN(i, bp_hiddens_in_layer(net,l))
            net->hiddens[l][i]->value_reprojected = 0;
    }

    /* set the neuron active */
    n = net->hiddens[layer][neuron_index];
    n->value_reprojected = 0.75f;

    bp_neuron_reproject(n);
    if (layer > 0) {
        /* apply the sigmoid function in the previous layer,
           as with feedforward */
        COUNTDOWN(i, bp_hiddens_in_layer(net,layer-1)) {
            n = net->hiddens[layer-1][i];
            n->value_reprojected =
                AF(n->value_reprojected);
        }
    }

    /* reproject through the hidden layers */
    for (int l = layer-1; l > 0; l--) {
        /* for every unit in the hidden layer */
        COUNTDOWN(i, bp_hiddens_in_layer(net,l))
            bp_neuron_reproject(net->hiddens[l][i]);

        /* apply the sigmoid function in the previous layer,
           as with feedforward */
        COUNTDOWN(i, bp_hiddens_in_layer(net,l-1)) {
            n = net->hiddens[l-1][i];
            n->value_reprojected =
                AF(n->value_reprojected);
        }
    }
}

/**
* @brief Adjust connection weights and bias values
* @param net Backprop neural net object
*/
void bp_learn(bp * net, int current_hidden_layer)
{
    int start_hidden_layer = current_hidden_layer-1;

    /* for each hidden layers */
    if (start_hidden_layer < 0)
        start_hidden_layer = 0;

    FOR(l, start_hidden_layer, net->HiddenLayers) {
        COUNTDOWN(i, bp_hiddens_in_layer(net,l))
            bp_neuron_learn(net->hiddens[l][i],net->learningRate);
    }

    COUNTDOWN(i, net->NoOfOutputs)
        bp_neuron_learn(net->outputs[i],net->learningRate);
}

/**
* @brief Set the value of an input
* @param net Backprop neural net object
* @param index The index number of the input unit
* @param value The value to set the unit to in the range 0.0 to 1.0
*/
void bp_set_input(bp * net, int index, float value)
{
    net->inputs[index]->value = value;
}

/**
* @brief Normalises the input values
* @param net Backprop neural net object
*/
void bp_normalise_inputs(bp * net)
{
    float max = 0,  min = 1;

    COUNTDOWN(i, net->NoOfInputs) {
        if (net->inputs[i]->value > max)
            max = net->inputs[i]->value;

        if (net->inputs[i]->value < min)
            min = net->inputs[i]->value;
    }

    float range = max - min;
    if (range > 0.00001f) {
        COUNTDOWN(i, net->NoOfInputs)
            net->inputs[i]->value =
                0.25f + ((net->inputs[i]->value-min)*0.5f/range);
    }
}

/**
* @brief Sets the inputs to a text string
* @param net Backprop neural net object
* @param text The text string
*/
void bp_set_input_text(bp * net, char * text)
{
    enc_text_to_binary(text, net->inputs, net->NoOfInputs, 0, strlen(text));
}

/**
* @brief Set the inputs of the network from a patch within an image.
*        It is assumed that the image is mono (1 byte per pixel)
* @param net Backprop neural net object
* @param img Array storing the image
* @param image_width Width of the image in pixels
* @param image_height Height of the image in pixels
* @param tx Top left x coordinate of the patch within the image
* @param ty Top left y coordinate of the patch within the image
*/
void bp_inputs_from_image_patch(bp * net,
                                unsigned char img[],
                                int image_width, int image_height,
                                int tx, int ty)
{
    int i = 0, idx;

    /* The patch size is calculated from the number of inputs
        of the neural net.  It's assumed to be square. */
    int patch_size = (int)sqrt(net->NoOfInputs);

    /* make sure that the patch fits within the number of inputs */
    assert(patch_size*patch_size <= net->NoOfInputs);

    /* set the inputs from the patch */
    FOR(py, ty, ty+patch_size) {
        if (py >= image_height) break;
        FOR(px, tx, tx+patch_size) {
            if (px >= image_width) break;

            /* array index within the image */
            idx = (py*image_width) + px;

            /* set the input value within the range 0.25 to 0.75 */
            bp_set_input(net, i, 0.25f + (img[idx]*0.5f/255.0f));
            i++;
        }
    }
}

/**
* @brief Set the inputs of the network from an image.
*        It is assumed that the image is mono (1 byte per pixel)
* @param net Backprop neural net object
* @param img Array storing the image
* @param image_width Width of the image in pixels
* @param image_height Height of the image in pixels
*/
void bp_inputs_from_image(bp * net,
                          unsigned char img[],
                          int image_width, int image_height)
{
    /* check that the number of inputs is the same as the
       number of pixels */
    assert(net->NoOfInputs == image_width*image_height);

    /* set the inputs */
    /* set the input value within the range 0.25 to 0.75 */
    COUNTDOWN(i, image_width*image_height)
        bp_set_input(net, i, 0.25f + (img[i]*0.5f/255.0f));
}

/**
* @brief Plots weight matrices within an image
* @param net Backprop neural net object
* @param filename Filename of the image to save as
* @param image_width Width of the image in pixels
* @param image_height Height of the image in pixels
* @param input_image_width When displaying all inputs as an image this
         is the number of inputs across.  Set this to zero for the
         inputs image to be square.
*/
int bp_plot_weights(bp * net,
                    char * filename,
                    int image_width, int image_height,
                    int input_image_width)
{
    int neurons_x, neurons_y, ty, by, h, ix, iy;
    int wx, wy, inputs_x, inputs_y, n, i, unit, no_of_neurons;
    int no_of_weights,wdth, max_unit;
    float neuronx, neurony, dw, db, min_bias, max_bias;
    float min_activation, max_activation, da;
    bp_neuron ** neurons, * curr_neuron;
    unsigned char * img;

    /* allocate memory for the image */
    img = (unsigned char*)malloc(image_width*image_height*3*
                                 sizeof(unsigned char));
    if (!img)
        return -1;

    /* clear the image with a white background */
    memset((void*)img, '\255',
           image_width*image_height*3*sizeof(unsigned char));

    /* dimension of the neurons matrix for each layer */
    neurons_x = (int)sqrt(net->NoOfHiddens);
    neurons_y = (net->NoOfHiddens/neurons_x);

    /* dimensions of the weight matrix */
    if (input_image_width <= 0)
        inputs_x = (int)sqrt(net->NoOfInputs);
    else
        inputs_x = input_image_width;

    inputs_y = (net->NoOfInputs/inputs_x);

    no_of_weights = net->NoOfInputs;

    /* plot the inputs */
    ty = 0;
    by = image_height/(net->HiddenLayers+3);
    h = (by-ty)*95/100;
    wdth = h;
    if (wdth>=image_width) wdth=image_width;
    COUNTUP(y, h) {
        iy = y*inputs_y/h;
        COUNTUP(x, wdth) {
            ix = x*inputs_x/wdth;
            unit = (iy*inputs_x) + ix;
            if (unit < net->NoOfInputs) {
                n = (y*image_width + x)*3;
                img[n] = (unsigned char)(net->inputs[unit]->value*255);
                img[n+1] = img[n];
                img[n+2] = img[n];
            }
        }
    }

    COUNTUP(layer, net->HiddenLayers+1) {
        /* vertical top and bottom coordinates */
        ty = (layer+1)*image_height/(net->HiddenLayers+3);
        by = (layer+2)*image_height/(net->HiddenLayers+3);
        h = (by-ty)*95/100;

        /* reset ranges */
        min_bias = 9999999.0f;
        max_bias = -9999999.0f;
        min_activation = 9999999.0f;
        max_activation = -9999999.0f;

        /* number of patches across and down for the final layer */
        if (layer == net->HiddenLayers) {
            neurons_x = (int)sqrt(net->NoOfOutputs);
            neurons_y = (net->NoOfOutputs/neurons_x);
            neurons = net->outputs;
            no_of_neurons = net->NoOfOutputs;
            max_unit = net->NoOfOutputs;
        }
        else {
            neurons_x = (int)sqrt(bp_hiddens_in_layer(net,layer));
            neurons_y = (bp_hiddens_in_layer(net,layer)/neurons_x);
            neurons = net->hiddens[layer];
            no_of_neurons = bp_hiddens_in_layer(net,layer);
            max_unit = bp_hiddens_in_layer(net,layer);
        }

        /* get the bias range within this layer */
        COUNTUP(y, max_unit) {
            if (neurons[y]->bias < min_bias) min_bias = neurons[y]->bias;
            if (neurons[y]->bias > max_bias) max_bias = neurons[y]->bias;
            if (neurons[y]->value < min_activation)
                min_activation = neurons[y]->value;
            if (neurons[y]->value > max_activation)
                max_activation = neurons[y]->value;
        }

        /* update ranges */
        db = max_bias - min_bias;
        da = max_activation - min_activation;

        /* for every pixel within the region */
        FOR(y, ty, by) {
            neurony = (y-ty)*neurons_y/(float)h;
            /* y coordinate within the weights */
            wy = (neurony - (int)neurony)*inputs_y;
            COUNTUP(x, image_width) {
                neuronx = x*neurons_x/(float)image_width;
                /* x coordinate within the weights */
                wx = (neuronx - (int)neuronx)*inputs_x;
                /* coordinate within the image */
                n = ((y * image_width) + x)*3;
                /* weight index */
                i = (wy*inputs_x) + wx;
                if (i < no_of_weights) {
                    /* neuron index */
                    unit = ((int)neurony*neurons_x) + (int)neuronx;
                    if (unit < no_of_neurons)  {
                        curr_neuron = neurons[unit];
                        dw = curr_neuron->max_weight -
                            curr_neuron->min_weight;
                        if (dw > 0.0001f) {
                            img[n] =
                                (int)((curr_neuron->weights[i] -
                                       curr_neuron->min_weight)*255/dw);
                            img[n+1] =
                                (int)((curr_neuron->bias - min_bias)*255/db);
                            img[n+2] =
                                (int)((curr_neuron->value - min_activation)*
                                      255/da);
                        }
                        else {
                            img[n] =
                                (int)(curr_neuron->weights[i]*255);
                            img[n+1] = img[n];
                            img[n+2] = img[n];
                        }
                    }
                }
            }
        }
        if (layer < net->HiddenLayers) {
            /* dimensions of the weight matrix for the next layer */
            inputs_x = (int)sqrt(bp_hiddens_in_layer(net,layer));
            inputs_y = (bp_hiddens_in_layer(net,layer)/inputs_x);
            no_of_weights = bp_hiddens_in_layer(net,layer);
        }
    }

    ty = (net->HiddenLayers+2)*image_height/(net->HiddenLayers+3);
    by = (net->HiddenLayers+3)*image_height/(net->HiddenLayers+3);
    h = (by-ty)*95/100;

    inputs_x = (int)sqrt(net->NoOfOutputs);
    inputs_y = (net->NoOfOutputs/inputs_x);

    wdth = h;
    if (wdth >= image_width) wdth = image_width;
    COUNTUP(y, h) {
        iy = y*inputs_y/h;
        COUNTUP(x, wdth) {
            ix = x*inputs_x/wdth;
            unit = (iy*inputs_x) + ix;
            if (unit < net->NoOfOutputs) {
                n = ((ty+y)*image_width + x)*3;
                img[n] = (unsigned char)(net->outputs[unit]->value*255);
                img[n+1] = img[n];
                img[n+2] = img[n];
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

/**
* @brief Returns the value of one of the input units
* @param net Backprop neural net object
* @param index Index of the input unit
* @return value in the range 0.0 to 1.0
*/
float bp_get_input(bp * net, int index)
{
    return net->inputs[index]->value;
}

/**
* @brief Sets the value of one of the output units
* @param net Backprop neural net object
* @param index Index of the output unit
* @param value The value to set the output to in the range 0.0 to 1.0
*/
void bp_set_output(bp * net, int index, float value)
{
    net->outputs[index]->desiredValue = value;
}

/**
* @brief Gets the value of one of the hidden units
* @param net Backprop neural net object
* @param layer Index number of the hidden layer
* @param index Index of the unit within the given hidden layer
* @return value in the range 0.0 to 1.0
*/
float bp_get_hidden(bp * net, int layer, int index)
{
    return net->hiddens[layer][index]->value;
}

/**
* @brief Gets the value of one of the output units
* @param net Backprop neural net object
* @param index Index of the unit within the output layer
* @return value in the range 0.0 to 1.0
*/
float bp_get_output(bp * net, int index)
{
    return net->outputs[index]->value;
}

/**
* @brief Exclusion flags indicate that a unit has temporarily dropped out.
*        This clears the all the exclusion flags
* @param net Backprop neural net object
*/
static void bp_clear_dropouts(bp * net)
{
    /* if no dropouts then don't continue */
    if (net->DropoutPercent == 0) return;

    /* for every hidden layer */
    COUNTDOWN(l, net->HiddenLayers) {
        COUNTDOWN(i, bp_hiddens_in_layer(net,l))
            net->hiddens[l][i]->excluded = 0;
    }
}

/**
* @brief Randomly sets exclusion flags to cause units to drop out
* @param net Backprop neural net object
*/
static void bp_dropouts(bp * net)
{
    int no_of_dropouts, hidden_units=0;

    if (net->DropoutPercent == 0) return;

    /* total number of hidden units */
    COUNTDOWN(l, net->HiddenLayers)
        hidden_units += bp_hiddens_in_layer(net,l);

    /* total number of dropouts */
    no_of_dropouts = net->DropoutPercent*hidden_units/100;

    /* set the exclusion flags */
    COUNTDOWN(n, no_of_dropouts) {
        int l = rand_num(&net->random_seed)%net->HiddenLayers;
        int i = rand_num(&net->random_seed)%bp_hiddens_in_layer(net,l);
        net->hiddens[l][i]->excluded = 1;
    }
}

/**
* @brief Update the neural net during training
* @param net Backprop neural net object
*/
void bp_update(bp * net, int current_hidden_layer)
{
    bp_dropouts(net);
    bp_feed_forward(net);
    bp_backprop(net, current_hidden_layer);
    bp_learn(net, current_hidden_layer);
    bp_clear_dropouts(net);
}

/**
* @brief Save a neural network to file
* @brief fp File pointer
* @param net Backprop neural net object
* @return zero on success
*/
int bp_save(FILE * fp, bp * net)
{
    if (UINTWRITE(net->itterations) == 0)
        return -1;

    if (INTWRITE(net->NoOfInputs) == 0)
        return -2;

    if (INTWRITE(net->NoOfHiddens) == 0)
        return -3;

    if (INTWRITE(net->NoOfOutputs) == 0)
        return -4;

    if (INTWRITE(net->HiddenLayers) == 0)
        return -5;

    if (FLOATWRITE(net->learningRate) == 0)
        return -6;

    if (FLOATWRITE(net->noise) == 0)
        return -7;

    if (FLOATWRITE(net->BPerrorAverage) == 0)
        return -8;

    if (FLOATWRITE(net->DropoutPercent) == 0)
        return -9;

    COUNTUP(l, net->HiddenLayers) {
        COUNTUP(i, bp_hiddens_in_layer(net,l))
            bp_neuron_save(fp,net->hiddens[l][i]);
    }

    COUNTUP(i, net->NoOfOutputs)
        bp_neuron_save(fp,net->outputs[i]);

    return 0;
}

/**
* @brief Load a network from file
* @brief fp File pointer
* @param net Backprop neural net object
* @param random_seed Random number generator seed
* @returns zero on success
*/
int bp_load(FILE * fp, bp * net,
            unsigned int * random_seed)
{
    int no_of_inputs=0, no_of_hiddens=0, no_of_outputs=0;
    int hidden_layers=0;
    float learning_rate=0, noise=0, BPerrorAverage=0;
    float DropoutPercent=0;
    unsigned int itterations=0;

    if (UINTREAD(itterations) == 0)
        return -1;

    if (INTREAD(no_of_inputs) == 0)
        return -2;

    if (INTREAD(no_of_hiddens) == 0)
        return -3;

    if (INTREAD(no_of_outputs) == 0)
        return -4;

    if (INTREAD(hidden_layers) == 0)
        return -5;

    if (FLOATREAD(learning_rate) == 0)
        return -6;

    if (FLOATREAD(noise) == 0)
        return -7;

    if (FLOATREAD(BPerrorAverage) == 0)
        return -8;

    if (FLOATREAD(DropoutPercent) == 0)
        return -9;

    if (bp_init(net, no_of_inputs, no_of_hiddens,
                hidden_layers, no_of_outputs,
                random_seed) != 0)
        return -10;

    COUNTUP(l, net->HiddenLayers) {
        COUNTUP(i, bp_hiddens_in_layer(net,l)) {
            if (bp_neuron_load(fp,net->hiddens[l][i]) != 0)
                return -11;
        }
    }

    COUNTUP(i, net->NoOfOutputs) {
        if (bp_neuron_load(fp,net->outputs[i]) != 0)
            return -12;
    }

    net->learningRate = learning_rate;
    net->noise = noise;
    net->BPerrorAverage = BPerrorAverage;
    net->BPerror = BPerrorAverage;
    net->BPerrorTotal = BPerrorAverage;
    net->itterations = itterations;
    net->DropoutPercent = DropoutPercent;

    return 0;
}

/**
* @brief compares two networks and returns a greater than
*        zero value if they are the same
* @param net1 The first backprop neural net object
* @param net2 The second backprop neural net object
*/
int bp_compare(bp * net1, bp * net2)
{
    int retval;

    if (net1->NoOfInputs != net2->NoOfInputs)
        return -1;

    if (net1->NoOfHiddens != net2->NoOfHiddens)
        return -2;

    if (net1->NoOfOutputs != net2->NoOfOutputs)
        return -3;

    if (net1->HiddenLayers != net2->HiddenLayers)
        return -4;

    if (net1->learningRate != net2->learningRate)
        return -5;

    if (net1->noise != net2->noise)
        return -6;

    COUNTDOWN(l, net1->HiddenLayers) {
        COUNTDOWN(i, bp_hiddens_in_layer(net1,l)) {
            retval =
                bp_neuron_compare(net1->hiddens[l][i],
                                  net2->hiddens[l][i]);
            if (retval == 0)
                return -7;
        }
    }

    COUNTDOWN(i, net1->NoOfOutputs) {
        retval = bp_neuron_compare(net1->outputs[i], net2->outputs[i]);
        if (retval == 0)
            return -8;
    }

    if (net1->itterations != net2->itterations)
        return -9;

    if (net1->BPerrorAverage != net2->BPerrorAverage)
        return -9;

    if (net1->DropoutPercent!= net2->DropoutPercent)
        return -10;

    return 1;
}

/**
* @brief Extract the classification string from the filename.
*        This assumes a filename of the type class.instance.extension
* @param filename The filename to examine
* @param classification The returned classification
*/
void bp_get_classification_from_filename(char * filename,
                                         char * classification)
{
    int j, start=0;

    /* start with an empty classification string */
    classification[0] = 0;

    /* find the last separator */
    COUNTUP(i, strlen(filename)) {
        if (filename[i] == '/')
            start = i+1;
    }

    /* find the first full stop */
    for (j = start; j < strlen(filename); j++) {
        if ((filename[j] == '.') ||
            (filename[j] == '-') ||
            (filename[j] == '_')) break;
        classification[j-start] = filename[j];
    }

    /* add a string terminator */
    classification[j-start] = 0;
}

/**
* @brief Takes a set of classification text descriptions (labels) for
*        each instance within a training or test set and produces an
*        array of classification numbers corresponding to the text
*        descriptions. It's easier for the system to deal with
*        classification numbers rather than text descriptions.
* @param no_of_instances The number of instances in the training or test set
* @param instance_classification Text Description for each instance
* @param numbers Array of numbers corresponding to each instance
*/
int bp_classifications_to_numbers(int no_of_instances,
                                  char ** instance_classification,
                                  int * numbers)
{
    int j;
    int unique_ctr = 0;
    char ** unique_classification;

    /* allocate memory for a list of unique descriptions (labels) */
    unique_classification =
        (char**)malloc(no_of_instances * sizeof(char*));

    if (!unique_classification)
        return -1;

    /* create a list of unique classification names */
    COUNTUP(i, no_of_instances) {
        /* for every class number assigned so far */
        for (j = 0; j < unique_ctr; j++) {
            /* is this instance description (label) the same as a previous
                instance description ? */
            if (strcmp(instance_classification[i],
                       unique_classification[j])==0) {
                /* assign the same class number and finish the search */
                numbers[i] = j;
                break;
            }
        }
        /* if this instance has a description which has not been used before */
        if (j == unique_ctr) {
            /* store the description */
            unique_classification[unique_ctr] =
                (char*)malloc((1+strlen(instance_classification[i]))*
                              sizeof(char));
            if (!unique_classification[unique_ctr])
                return -2;

            sprintf(unique_classification[unique_ctr],
                    "%s", instance_classification[i]);

            /* store the classification number */
            numbers[i] = unique_ctr;

            /* increment the number of classes */
            unique_ctr++;
        }
    }

    /* free memory which was used to store descriptions */
    COUNTDOWN(i, unique_ctr)
        free(unique_classification[i]);

    free(unique_classification);
    return 0;
}
