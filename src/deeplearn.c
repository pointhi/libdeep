/*
  libdeep - a library for deep learning
  Copyright (C) 2013-2015  Bob Mottram <bob@robotics.uk.to>

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

#include "deeplearn.h"

/**
 * @brief Returns a training error threshold for the given layer
 * @param learner Deep learner object
 * @param index Layer index
 * @returns Training error threshold (percentage value)
 */
float deeplearn_get_error_threshold(deeplearn * learner, int index)
{
    return(learner->error_threshold[index]);
}

/**
 * @brief Sets a training error threshold
 * @param learner Deep learner object
 * @param index Layer index
 * @param value Threshold value as a percentage
 */
void deeplearn_set_error_threshold(deeplearn * learner, int index, float value)
{
    learner->error_threshold[index] = value;
}

/**
 * @brief Update the learning history
 * @param learner Deep learner object
 */
static void deeplearn_update_history(deeplearn * learner)
{
    int i;
    float error_value;

    if (learner->history_step == 0) return;

    learner->history_ctr++;
    if (learner->history_ctr >= learner->history_step) {
        error_value = learner->BPerror;
        if (error_value == DEEPLEARN_UNKNOWN_ERROR) {
            error_value = 0;
        }

        learner->history[learner->history_index] =
            error_value;
        learner->history_index++;
        learner->history_ctr = 0;

        if (learner->history_index >= DEEPLEARN_HISTORY_SIZE) {
            for (i = 0; i < learner->history_index; i++) {
                learner->history[i/2] = learner->history[i];
            }
            learner->history_index /= 2;
            learner->history_step *= 2;
        }
    }
}

/**
 * @brief Initialise a deep learner
 * @param learner Deep learner object
 * @param no_of_inputs The number of input fields
 * @param no_of_hiddens The number of hidden units within each layer
 * @param hidden_layers The number of hidden layers
 * @param no_of_outputs The number of output units
 * @param error_threshold Minimum training error for each hidden layer plus
 *        the output layer
 * @param random_seed Random number generator seed
 */
int deeplearn_init(deeplearn * learner,
                   int no_of_inputs,
                   int no_of_hiddens,
                   int hidden_layers,
                   int no_of_outputs,
                   float error_threshold[],
                   unsigned int * random_seed)
{
    int i;

    /* no training/test data yet */
    learner->data = 0;
    learner->data_samples = 0;
    learner->indexed_data = 0;
    learner->indexed_data_samples = 0;

    learner->training_data = 0;
    learner->training_data_samples = 0;
    learner->indexed_training_data = 0;
    learner->indexed_training_data_samples = 0;

    learner->training_data_labeled = 0;
    learner->training_data_labeled_samples = 0;
    learner->indexed_training_data_labeled = 0;
    learner->indexed_training_data_labeled_samples = 0;

    learner->test_data = 0;
    learner->test_data_samples = 0;
    learner->indexed_test_data = 0;
    learner->indexed_test_data_samples = 0;

    learner->no_of_input_fields = 0;
    learner->field_length = 0;

    learner->training_ctr = 0;
    learner->history_plot_interval = 100000;
    sprintf(learner->history_plot_filename,"%s","training.png");
    sprintf(learner->history_plot_title,"%s","Training History");

    learner->input_range_min = (float*)malloc(no_of_inputs*sizeof(float));
    if (!learner->input_range_min) {
        return -1;
    }
    learner->input_range_max = (float*)malloc(no_of_inputs*sizeof(float));
    if (!learner->input_range_max) {
        return -2;
    }
    learner->output_range_min = (float*)malloc(no_of_outputs*sizeof(float));
    if (!learner->output_range_min) {
        return -3;
    }
    learner->output_range_max = (float*)malloc(no_of_outputs*sizeof(float));
    if (!learner->output_range_max) {
        return -4;
    }

    for (i = 0; i < no_of_inputs; i++) {
        learner->input_range_min[i] = 99999;
        learner->input_range_max[i] = -99999;
    }
    for (i = 0; i < no_of_outputs; i++) {
        learner->output_range_min[i] = 99999;
        learner->output_range_max[i] = -99999;
    }

    /* has not been trained */
    learner->training_complete = 0;

    /* create the error thresholds for each layer */
    learner->error_threshold =
        (float*)malloc((hidden_layers+1)*sizeof(float));
    if (!learner->error_threshold) {
        return -5;
    }
    memcpy((void*)learner->error_threshold,
           (void*)error_threshold,
           (hidden_layers+1)*sizeof(float));

    /* clear history */
    learner->history_index = 0;
    learner->history_ctr = 0;
    learner->history_step = 1;

    /* clear the number of training itterations */
    learner->itterations = 0;

    /* set the current layer being trained */
    learner->current_hidden_layer = 0;

    /* create the network */
    learner->net = (bp*)malloc(sizeof(bp));
    if (!learner->net) {
        return -6;
    }

    /* initialise the network */
    if (bp_init(learner->net,
                no_of_inputs, no_of_hiddens,
                hidden_layers, no_of_outputs,
                random_seed) != 0) {
        return -7;
    }

    /* create the autocoder */
    learner->autocoder = (ac**)malloc(sizeof(ac*)*hidden_layers);
    if (!learner->autocoder) {
        return -8;
    }
    for (i = 0; i < hidden_layers; i++) {
        learner->autocoder[i] = (ac*)malloc(sizeof(ac));
        if (!learner->autocoder[i]) {
            return -9;
        }
        if (i == 0) {
            /* if this is the first hidden layer then number of inputs
               for the autocoder is the same as the number of
               neural net input units */
            if (autocoder_init(learner->autocoder[i], no_of_inputs,
                               bp_hiddens_in_layer(learner->net,i),
                               learner->net->random_seed) != 0) {
                return -10;
            }
        }
        else {
            if (autocoder_init(learner->autocoder[i],
                               bp_hiddens_in_layer(learner->net,i-1),
                               bp_hiddens_in_layer(learner->net,i),
                               learner->net->random_seed) != 0) {
                return -11;
            }
        }

    }

    learner->BPerror = DEEPLEARN_UNKNOWN_ERROR;
    return 0;
}

/**
 * @brief Feeds the input values through the network towards the outputs
 * @param learner Deep learner object
 */
void deeplearn_feed_forward(deeplearn * learner)
{
    bp_feed_forward(learner->net);
}

/**
 * @brief Returns true if currently training the final layer
 * @param learner Deep learner object
 * @returns True if training the last layer
 */
int deeplearn_training_last_layer(deeplearn * learner)
{
    return (learner->current_hidden_layer >=
            learner->net->HiddenLayers);
}

/**
 * @brief Copy autocoder hidden units to a hidden layer of the deep learner
 * @param learner Deep learner object
 * @param hidden_layer Index of the layer to be copied
 */
void copy_autocoder_to_hidden_layer(deeplearn * learner, int hidden_layer)
{
    ac * autocoder = learner->autocoder[hidden_layer];
    /* for each unit on the hidden layer */
    for (int i = 0; i < bp_hiddens_in_layer(learner->net,hidden_layer); i++) {
        bp_neuron * nrn  = learner->net->hiddens[hidden_layer][i];
        nrn->bias = autocoder->bias[i];
        memcpy((void*)nrn->weights,
               &autocoder->weights[i*autocoder->NoOfInputs],
               autocoder->NoOfInputs*sizeof(float));
    }
}

/**
 * @brief Trains the autocoder for a given hidden layer
 * @param net Backprop object
 * @param autocoder Autocoder object
 * @param current_layer Index of the current hidden layer
 */
void deeplearn_pretrain(bp * net, ac * autocoder, int current_layer)
{
    bp_feed_forward_layers(net, current_layer);
    if (current_layer > 0) {
        /* copy the hidden unit values to the inputs
           of the autocoder */
        for (int i = 0; i < bp_hiddens_in_layer(net,current_layer-1); i++) {
            float hidden_value = bp_get_hidden(net, current_layer-1, i);
            autocoder_set_input(autocoder, i, hidden_value);
        }
    }
    else {
        /* copy the input unit values to the inputs
           of the autocoder */
        for (int i = 0; i < net->NoOfInputs; i++) {
            autocoder_set_input(autocoder, i,
                                bp_get_input(net, i));
        }
    }
    autocoder_update(autocoder);
}

/**
 * @brief Performs training initially using autocoders
 *        for each hidden
 *        layer and eventually for the entire network.
 * @param learner Deep learner object
 */
void deeplearn_update(deeplearn * learner)
{
    float minimum_error_percent = 0;
    int current_layer = learner->current_hidden_layer;

    /* only continue if training is not complete */
    if (learner->training_complete == 1) return;

    /* get the maximum backprop error after which a layer
       will be considered to have been trained */
    minimum_error_percent =
        learner->error_threshold[current_layer];

    /* If there is only a single hidden layer */
    if ((current_layer == 0) &&
        (learner->net->HiddenLayers == 1)) {
        current_layer = 1;
        learner->current_hidden_layer = current_layer;
    }

    /* pretraining of autocoders */
    if (current_layer < learner->net->HiddenLayers) {

        /* train the autocoder for this layer */
        deeplearn_pretrain(learner->net,
                           learner->autocoder[current_layer],
                           current_layer);

        /* update the backprop error value from the autocoder */
        learner->BPerror =
            learner->autocoder[current_layer]->BPerrorPercent;

        /* If below the error threshold.
           Only do this after a minimum number of itterations
           in order to allow the running average to stabilise */
        if ((learner->BPerror != DEEPLEARN_UNKNOWN_ERROR) &&
            (learner->BPerror < minimum_error_percent) &&
            (learner->autocoder[current_layer]->itterations > 100)) {

            copy_autocoder_to_hidden_layer(learner, current_layer);

            /* advance to the next hidden layer */
            learner->current_hidden_layer++;

            /* reset the error value */
            learner->BPerror = DEEPLEARN_UNKNOWN_ERROR;
        }
    }
    else {
        bp_update(learner->net,0);

        /* update the backprop error value */
        learner->BPerror = learner->net->BPerrorPercent;

        /* set the training completed flag */
        if (learner->BPerror < minimum_error_percent) {
            learner->training_complete = 1;
        }
    }

    /* record the history of error values */
    deeplearn_update_history(learner);

    /* increment the number of itterations */
    if (learner->net->itterations < UINT_MAX) {
        learner->net->itterations++;
    }
}

/**
 * @brief Perform continuous unsupervised learning
 * @param learner deep learner object
 */
void deeplearn_update_continuous(deeplearn * learner)
{
    int i;

    learner->BPerror = 0;

    for (i = 0; i < learner->net->HiddenLayers; i++) {
        /* train the autocoder for this layer */
        deeplearn_pretrain(learner->net, learner->autocoder[i], i);

        /* update the backprop error value */
        learner->BPerror += learner->autocoder[i]->BPerrorPercent;

        copy_autocoder_to_hidden_layer(learner, i);
    }

    learner->BPerror /= learner->net->HiddenLayers;

    /* record the history of error values */
    deeplearn_update_history(learner);
}

/**
 * @brief Deallocates memory for the given deep learner object
 * @param learner Deep learner object
 */
void deeplearn_free(deeplearn * learner)
{
    /* clear any data */
    deeplearndata * sample = learner->data;
    deeplearndata * prev_sample;
    int i;

    free(learner->input_range_min);
    free(learner->input_range_max);
    free(learner->output_range_min);
    free(learner->output_range_max);
    if (learner->field_length != 0) {
        free(learner->field_length);
    }

    while (sample != 0) {
        prev_sample = sample;
        sample = (deeplearndata *)sample->next;
        if (prev_sample != 0) {
            if (prev_sample->inputs_text != 0) {
                /* clear any input text strings */
                for (i = 0; i < learner->no_of_input_fields; i++) {
                    if (prev_sample->inputs_text[i] != 0) {
                        free(prev_sample->inputs_text[i]);
                    }
                }
                free(prev_sample->inputs_text);
            }
            /* clear numerical fields */
            free(prev_sample->inputs);
            free(prev_sample->outputs);
            free(prev_sample);
        }
    }

    /* free training samples */
    deeplearndata_meta * training_sample = learner->training_data;
    deeplearndata_meta * prev_training_sample;
    while (training_sample != 0) {
        prev_training_sample = training_sample;
        training_sample = (deeplearndata_meta *)training_sample->next;
        free(prev_training_sample);
    }

    /* free labeled training samples */
    deeplearndata_meta * training_sample_labeled = learner->training_data_labeled;
    deeplearndata_meta * prev_training_sample_labeled;
    while (training_sample_labeled != 0) {
        prev_training_sample_labeled = training_sample_labeled;
        training_sample_labeled = (deeplearndata_meta *)training_sample_labeled->next;
        free(prev_training_sample_labeled);
    }

    /* free test samples */
    deeplearndata_meta * test_sample = learner->test_data;
    deeplearndata_meta * prev_test_sample;
    while (test_sample != 0) {
        prev_test_sample = test_sample;
        test_sample = (deeplearndata_meta *)test_sample->next;
        free(prev_test_sample);
    }

    /* free the error thresholds */
    free(learner->error_threshold);

    /* free the autocoder */
    for (int i = 0; i < learner->net->HiddenLayers; i++) {
        autocoder_free(learner->autocoder[i]);
        free(learner->autocoder[i]);
        learner->autocoder[i] = 0;
    }
    free(learner->autocoder);

    /* free the learner */
    bp_free(learner->net);
    free(learner->net);
}

/**
 * @brief Sets the value of an input to the network
 * @param learner Deep learner object
 * @param index Index number of the input unit
 * @param value Value to set the input unit to in the range 0.0 to 1.0
 */
void deeplearn_set_input(deeplearn * learner, int index, float value)
{
    bp_set_input(learner->net, index, value);
}

/**
 * @brief Sets the inputs to a text string
 *        Note that this sets the entire inputs, not a field
 * @param learner Deep learner object
 * @param text The text string
 */
void deeplearn_set_input_text(deeplearn * learner, char * text)
{
    bp_set_input_text(learner->net, text);
}

/**
 * @brief Sets inputs from the given data sample.
 *        The sample can contain arbitrary floating point values, so these
 *        need to be normalised into a 0.25-0.75 range
 * @param learner Deep learner object
 */
void deeplearn_set_inputs(deeplearn * learner, deeplearndata * sample)
{
    float value, range, normalised;
    int pos = 0;

    for (int i = 0; i < learner->no_of_input_fields; i++) {
        if (learner->field_length[i] > 0) {
            /* text value */
            enc_text_to_binary(sample->inputs_text[i],
                               learner->net->inputs,
                               learner->net->NoOfInputs,
                               pos, learner->field_length[i]/CHAR_BITS);
            pos += learner->field_length[i];
        }
        else {
            /* numerical */
            value = sample->inputs[i];
            range = learner->input_range_max[i] - learner->input_range_min[i];
            if (range > 0) {
                normalised = (((value - learner->input_range_min[i])/range)*0.5) + 0.25;
                deeplearn_set_input(learner, pos, normalised);
            }
            pos++;
        }
    }
}

/**
 * @brief Sets a numeric value for the given input field
 * @param learner Deep learner object
 * @param fieldindex Index number of the input field.
 *        This is not necessarily the same as the input index
 * @param value Value to set the input unit to in the range 0.0 to 1.0
 * @returns zero on success
 */
int deeplearn_set_input_field(deeplearn * learner, int fieldindex, float value)
{
    int i, pos=0;

    if (learner->no_of_input_fields == 0) {
        /* No fields are defined
           Assume you meant fieldindex to be the input index */
        bp_set_input(learner->net, fieldindex, value);
        return -1;
    }

    if (learner->field_length == 0) {
        /* no field length array defined */
        return -2;
    }

    if (learner->field_length[fieldindex] > 0) {
        /* this is a text field */
        return -3;
    }

    /* get the offset (first input unit index) of the input field */
    for (i = 0; i < fieldindex; i++) {
        if (learner->field_length[i] == 0) {
            pos++;
        }
        else {
            pos += learner->field_length[i];
        }
    }

    /* set the value */
    bp_set_input(learner->net, pos, value);
    return 0;
}

/**
 * @brief Sets a text value for the given input field
 * @param learner Deep learner object
 * @param fieldindex Index number of the input field.
 *        This is not necessarily the same as the input index
 * @param text Text value for the field
 * @returns zero on success
 */
int deeplearn_set_input_field_text(deeplearn * learner, int fieldindex, char * text)
{
    int i, pos=0;

    if (learner->no_of_input_fields == 0) {
        /* No fields are defined */
        return -1;
    }

    if (learner->field_length == 0) {
        /* no field length array defined */
        return -2;
    }

    if (learner->field_length[fieldindex] == 0) {
        /* this is a text field */
        return -3;
    }

    /* get the offset (first input unit index) of the input field */
    for (i = 0; i < fieldindex; i++) {
        if (learner->field_length[i] == 0) {
            pos++;
        }
        else {
            pos += learner->field_length[i];
        }
    }

    /* set the value */
    enc_text_to_binary(text,
                       learner->net->inputs,
                       learner->net->NoOfInputs, pos,
                       learner->field_length[fieldindex]/CHAR_BITS);
    return 0;
}

/**
 * @brief Sets the value of an output unit
 * @param learner Deep learner object
 * @param index Index number of the output unit
 * @param value Value to set the output unit to in the range 0.0 to 1.0
 */
void deeplearn_set_output(deeplearn * learner, int index, float value)
{
    bp_set_output(learner->net, index, value);
}

/**
 * @brief Sets outputs from the given data sample.
 *        The sample can contain arbitrary floating point values, so these
 *        need to be normalised into a 0.25-0.75 range
 * @param learner Deep learner object
 * @param sample The data sample from which to obtain the output values
 */
void deeplearn_set_outputs(deeplearn * learner, deeplearndata * sample)
{
    for (int i = 0; i < learner->net->NoOfOutputs; i++) {
        float value = sample->outputs[i];
        float range = learner->output_range_max[i] - learner->output_range_min[i];
        if (range > 0) {
            float normalised = (((value - learner->output_range_min[i])/range)*0.5) + 0.25;
            deeplearn_set_output(learner, i, normalised);
        }
    }
}

/**
 * @brief Returns the values of outputs within their normal range
 * @param learner Deep learner object
 * @param outputs The returned output values
 */
void deeplearn_get_outputs(deeplearn * learner, float * outputs)
{
    for (int i = 0; i < learner->net->NoOfOutputs; i++) {
        float value = deeplearn_get_output(learner, i);
        float range = learner->output_range_max[i] - learner->output_range_min[i];
        if (range > 0) {
            outputs[i] =
                (((value - 0.25f)/0.5f)*range) + learner->output_range_min[i];
        }
    }
}

/**
 * @brief Returns the value of an output unit
 * @param learner Deep learner object
 * @param index Index number of the output unit
 * @return Value of the output unit to in the range 0.0 to 1.0
 */
float deeplearn_get_output(deeplearn * learner, int index)
{
    return bp_get_output(learner->net, index);
}

/**
 * @brief Gets the output class as an integer value
 * @param learner Deep learner object
 * @return output class
 */
int deeplearn_get_class(deeplearn * learner)
{
    int i, class = -9999;
    float max = -1;
    for (i = 0; i < learner->net->NoOfOutputs; i++) {
        if (bp_get_output(learner->net, i) > max) {
            max = bp_get_output(learner->net, i);
            class = i;
        }
    }
    return class;
}

/**
 * @brief Sets the output class
 * @param learner Deep learner object
 * @param class The class number
 */
void deeplearn_set_class(deeplearn * learner, int class)
{
    int i;
    for (i = 0; i < learner->net->NoOfOutputs; i++) {
        if (i != class) {
            bp_set_output(learner->net, i, 0.25f);
        }
        else {
            bp_set_output(learner->net, i, 0.75f);
        }
    }
}

/**
 * @brief Saves the given deep learner object to a file
 * @param fp File pointer
 * @param learner Deep learner object
 * @return zero value on success
 */
int deeplearn_save(FILE * fp, deeplearn * learner)
{
    if (fwrite(&learner->training_complete, sizeof(int), 1, fp) == 0) {
        return -1;
    }
    if (fwrite(&learner->itterations, sizeof(unsigned int), 1, fp) == 0) {
        return -2;
    }
    if (fwrite(&learner->current_hidden_layer, sizeof(int), 1, fp) == 0) {
        return -3;
    }
    if (fwrite(&learner->BPerror, sizeof(float), 1, fp) == 0) {
        return -4;
    }
    if (fwrite(&learner->no_of_input_fields, sizeof(int), 1, fp) == 0) {
        return -5;
    }
    if (learner->no_of_input_fields > 0) {
        if (fwrite(learner->field_length, sizeof(int),
                   learner->no_of_input_fields, fp) == 0) {
            return -6;
        }
    }

    if (bp_save(fp, learner->net) != 0) {
        return -7;
    }

    for (int i = 0; i < learner->net->HiddenLayers; i++) {
        if (autocoder_save(fp, learner->autocoder[i]) != 0) {
            return -8;
        }
    }

    /* save error thresholds */
    if (fwrite(learner->error_threshold, sizeof(float),
               learner->net->HiddenLayers+1, fp) == 0) {
        return -9;
    }

    /* save ranges */
    if (fwrite(learner->input_range_min, sizeof(float), learner->net->NoOfInputs, fp) == 0) {
        return -10;
    }
    if (fwrite(learner->input_range_max, sizeof(float), learner->net->NoOfInputs, fp) == 0) {
        return -11;
    }
    if (fwrite(learner->output_range_min, sizeof(float), learner->net->NoOfOutputs, fp) == 0) {
        return -12;
    }
    if (fwrite(learner->output_range_max, sizeof(float), learner->net->NoOfOutputs, fp) == 0) {
        return -13;
    }

    /* save the history */
    if (fwrite(&learner->history_index, sizeof(int), 1, fp) == 0) {
        return -14;
    }
    if (fwrite(&learner->history_ctr, sizeof(int), 1, fp) == 0) {
        return -15;
    }
    if (fwrite(&learner->history_step, sizeof(int), 1, fp) == 0) {
        return -16;
    }
    if (fwrite(learner->history, sizeof(float),
               learner->history_index, fp) == 0) {
        return -17;
    }

    return 0;
}

/**
 * @brief Loads a deep learner object from file
 * @param fp File pointer
 * @param learner Deep learner object
 * @param random_seed Random number generator seed
 * @return zero value on success
 */
int deeplearn_load(FILE * fp, deeplearn * learner,
                   unsigned int * random_seed)
{
    /* no training/test data yet */
    learner->data = 0;
    learner->data_samples = 0;
    learner->training_data = 0;
    learner->training_data_samples = 0;
    learner->training_data_labeled = 0;
    learner->training_data_labeled_samples = 0;
    learner->test_data = 0;
    learner->test_data_samples = 0;

    if (fread(&learner->training_complete, sizeof(int), 1, fp) == 0) {
        return -1;
    }
    if (fread(&learner->itterations, sizeof(unsigned int), 1, fp) == 0) {
        return -2;
    }
    if (fread(&learner->current_hidden_layer, sizeof(int), 1, fp) == 0) {
        return -3;
    }
    if (fread(&learner->BPerror, sizeof(float), 1, fp) == 0) {
        return -4;
    }
    if (fread(&learner->no_of_input_fields, sizeof(int), 1, fp) == 0) {
        return -5;
    }
    learner->field_length = 0;
    if (learner->no_of_input_fields > 0) {
        learner->field_length = (int*)malloc(learner->no_of_input_fields*sizeof(int));
        if (fread(learner->field_length, sizeof(int), learner->no_of_input_fields, fp) == 0) {
            return -6;
        }
    }

    learner->net = (bp*)malloc(sizeof(bp));
    if (bp_load(fp, learner->net, random_seed) != 0) {
        return -7;
    }

    learner->autocoder = (ac**)malloc(sizeof(ac*)*learner->net->HiddenLayers);
    if (!learner->autocoder) {
        return -8;
    }
    for (int i = 0; i < learner->net->HiddenLayers; i++) {
        learner->autocoder[i] = (ac*)malloc(sizeof(ac));
        if (autocoder_load(fp, learner->autocoder[i], 1) != 0) {
            return -9;
        }
    }

    /* load error thresholds */
    learner->error_threshold =
        (float*)malloc((learner->net->HiddenLayers+1)*
                       sizeof(float));
    if (fread(learner->error_threshold, sizeof(float),
              learner->net->HiddenLayers+1, fp) == 0) {
        return -10;
    }

    /* load ranges */
    learner->input_range_min = (float*)malloc(learner->net->NoOfInputs*sizeof(float));
    if (!learner->input_range_min) {
        return -15;
    }
    learner->input_range_max = (float*)malloc(learner->net->NoOfInputs*sizeof(float));
    if (!learner->input_range_max) {
        return -16;
    }
    learner->output_range_min = (float*)malloc(learner->net->NoOfOutputs*sizeof(float));
    if (!learner->output_range_min) {
        return -17;
    }
    learner->output_range_max = (float*)malloc(learner->net->NoOfOutputs*sizeof(float));
    if (!learner->output_range_max) {
        return -18;
    }
    if (fread(learner->input_range_min, sizeof(float), learner->net->NoOfInputs, fp) == 0) {
        return -19;
    }
    if (fread(learner->input_range_max, sizeof(float), learner->net->NoOfInputs, fp) == 0) {
        return -20;
    }
    if (fread(learner->output_range_min, sizeof(float), learner->net->NoOfOutputs, fp) == 0) {
        return -21;
    }
    if (fread(learner->output_range_max, sizeof(float), learner->net->NoOfOutputs, fp) == 0) {
        return -22;
    }

    /* load the history */
    if (fread(&learner->history_index, sizeof(int), 1, fp) == 0) {
        return -11;
    }
    if (fread(&learner->history_ctr, sizeof(int), 1, fp) == 0) {
        return -12;
    }
    if (fread(&learner->history_step, sizeof(int), 1, fp) == 0) {
        return -13;
    }
    if (fread(learner->history, sizeof(float),
              learner->history_index, fp) == 0) {
        return -14;
    }

    return 0;
}

/**
 * @brief Compares two deep learners and returns a greater
 *        than zero value if they are the same
 * @param learner1 First deep learner object
 * @param learner2 Second deep learner object
 * @return Greater than zero if the two learners are the same
 */
int deeplearn_compare(deeplearn * learner1,
                      deeplearn * learner2)
{
    int retval,i;

    if (learner1->current_hidden_layer !=
        learner2->current_hidden_layer) {
        return -1;
    }
    if (learner1->BPerror != learner2->BPerror) {
        return -2;
    }
    retval = bp_compare(learner1->net,learner2->net);
    if (retval < 1) return -3;
    if (learner1->history_index !=
        learner2->history_index) {
        return -5;
    }
    if (learner1->history_ctr !=
        learner2->history_ctr) {
        return -6;
    }
    if (learner1->history_step !=
        learner2->history_step) {
        return -7;
    }
    for (i = 0; i < learner1->history_index; i++) {
        if (learner1->history[i] !=
            learner2->history[i]) {
            return -8;
        }
    }
    if (learner1->itterations !=
        learner2->itterations) {
        return -9;
    }
    for (i = 0; i < learner1->net->HiddenLayers+1; i++) {
        if (learner1->error_threshold[i] !=
            learner2->error_threshold[i]) {
            return -10;
        }
    }
    for (i = 0; i < learner1->net->NoOfInputs; i++) {
        if (learner1->input_range_min[i] !=
            learner2->input_range_min[i]) {
            return -11;
        }
        if (learner1->input_range_max[i] !=
            learner2->input_range_max[i]) {
            return -12;
        }
    }
    for (i = 0; i < learner1->net->NoOfOutputs; i++) {
        if (learner1->output_range_min[i] !=
            learner2->output_range_min[i]) {
            return -13;
        }
        if (learner1->output_range_max[i] !=
            learner2->output_range_max[i]) {
            return -14;
        }
    }
    if (learner1->no_of_input_fields !=
        learner2->no_of_input_fields) {
        return -15;
    }
    return 1;
}

/**
 * @brief Uses gnuplot to plot the training error for the given learner
 * @param learner Deep learner object
 * @param filename Filename for the image to save as
 * @param title Title of the graph
 * @param image_width Width of the image in pixels
 * @param image_height Height of the image in pixels
 * @return zero on success
 */
int deeplearn_plot_history(deeplearn * learner,
                           char * filename, char * title,
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
    for (index = 0; index < learner->history_index; index++) {
        value = learner->history[index];
        fprintf(fp,"%d    %.10f\n",
                index*learner->history_step,value);
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
    fprintf(fp,"set title \"%s\"\n",title);
    fprintf(fp,"set xrange [0:%d]\n",
            learner->history_index*learner->history_step);
    fprintf(fp,"set yrange [0:%f]\n",max_value*102/100);
    fprintf(fp,"%s","set lmargin 9\n");
    fprintf(fp,"%s","set rmargin 2\n");
    fprintf(fp,"%s","set xlabel \"Time Step\"\n");
    fprintf(fp,"%s","set ylabel \"Training Error Percent\"\n");

    fprintf(fp,"%s","set grid\n");
    fprintf(fp,"%s","set key right top\n");

    fprintf(fp,"set terminal png size %d,%d\n",
            image_width, image_height);
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
 * @brief Updates the input units from a patch within a larger image
 * @param learner Deep learner object
 * @param img Image buffer (1 byte per pixel)
 * @param image_width Width of the image in pixels
 * @param image_height Height of the image in pixels
 * @param tx Top left x coordinate of the patch
 * @param ty Top left y coordinate of the patch
 */
void deeplearn_inputs_from_image_patch(deeplearn * learner,
                                       unsigned char * img,
                                       int image_width, int image_height,
                                       int tx, int ty)
{
    bp_inputs_from_image_patch(learner->net,
                               img, image_width, image_height,
                               tx, ty);
}

/**
 * @brief Updates the input units from an image
 * @param learner Deep learner object
 * @param img Image buffer (1 byte per pixel)
 * @param image_width Width of the image in pixels
 * @param image_height Height of the image in pixels
 */
void deeplearn_inputs_from_image(deeplearn * learner,
                                 unsigned char * img,
                                 int image_width, int image_height)
{
    bp_inputs_from_image(learner->net, img, image_width, image_height);
}

/**
 * @brief Sets the learning rate
 * @param learner Deep learner object
 * @param rate the learning rate in the range 0.0 to 1.0
 */
void deeplearn_set_learning_rate(deeplearn * learner, float rate)
{
    learner->net->learningRate = rate;

    for (int i = 0; i < learner->net->HiddenLayers; i++) {
        learner->autocoder[i]->learningRate = rate;
    }
}

/**
 * @brief Sets the percentage of units which drop out during training
 * @param learner Deep learner object
 * @param dropout_percent Percentage of units which drop out in the range 0 to 100
 */
void deeplearn_set_dropouts(deeplearn * learner, float dropout_percent)
{
    learner->net->DropoutPercent = dropout_percent;

    for (int i = 0; i < learner->net->HiddenLayers; i++) {
        learner->autocoder[i]->DropoutPercent = dropout_percent;
    }
}

/**
 * @brief Exports a trained network as a standalone C program
 * @param learner Deep learner object
 * @param filename The C source file to be produced
 * @returns zero on success
 */
static int deeplearn_export_c(deeplearn * learner, char * filename)
{
    FILE * fp;
    int i, j, k, no_of_weights;

    fp = fopen(filename,"w");
    if (!fp) {
        return -1;
    }

    fprintf(fp,"%s\n", "#include <stdio.h>");
    fprintf(fp,"%s\n", "#include <stdlib.h>");
    if (learner->no_of_input_fields > 0) {
        fprintf(fp,"%s\n", "#include <string.h>");
    }
    fprintf(fp,"%s\n\n", "#include <math.h>");

    if (learner->no_of_input_fields > 0) {
        fprintf(fp, "const int no_of_input_fields = %d;\n",
                learner->no_of_input_fields);
    }
    fprintf(fp, "const int no_of_inputs = %d;\n",
            learner->net->NoOfInputs);
    fprintf(fp, "const int no_of_hiddens = %d;\n",
            learner->net->NoOfHiddens);
    fprintf(fp, "const int no_of_outputs = %d;\n",
            learner->net->NoOfOutputs);
    fprintf(fp, "const int hidden_layers = %d;\n\n",
            learner->net->HiddenLayers);

    /* field lengths */
    if (learner->field_length != 0) {
        if (learner->no_of_input_fields > 0) {
            fprintf(fp, "%s", "int field_length[] = {\n");
            fprintf(fp, "%s", "  ");
            for (i = 0; i < learner->no_of_input_fields; i++) {
                fprintf(fp, "%d", learner->field_length[i]);
                if (i < learner->no_of_input_fields-1) {
                    fprintf(fp, ",");
                }
            }
            fprintf(fp, "%s", "\n};\n\n");
        }
    }

    /* ranges */
    fprintf(fp, "%s", "float input_range_min[] = {\n");
    fprintf(fp, "%s", "  ");
    for (i = 0; i < learner->net->NoOfInputs; i++) {
        fprintf(fp, "%.10f", learner->input_range_min[i]);
        if (i < learner->net->NoOfInputs-1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "%s", "\n};\n\n");
    fprintf(fp, "%s", "float input_range_max[] = {\n");
    fprintf(fp, "%s", "  ");
    for (i = 0; i < learner->net->NoOfInputs; i++) {
        fprintf(fp, "%.10f", learner->input_range_max[i]);
        if (i < learner->net->NoOfInputs-1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "%s", "\n};\n\n");
    fprintf(fp, "%s", "float output_range_min[] = {\n");
    fprintf(fp, "%s", "  ");
    for (i = 0; i < learner->net->NoOfOutputs; i++) {
        fprintf(fp, "%.10f", learner->output_range_min[i]);
        if (i < learner->net->NoOfOutputs-1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "%s", "\n};\n\n");
    fprintf(fp, "%s", "float output_range_max[] = {\n");
    fprintf(fp, "%s", "  ");
    for (i = 0; i < learner->net->NoOfOutputs; i++) {
        fprintf(fp, "%.10f", learner->output_range_max[i]);
        if (i < learner->net->NoOfOutputs-1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "%s", "\n};\n\n");

    /* hidden unit weights */
    for (i = 0; i < learner->net->HiddenLayers; i++) {
        fprintf(fp,
                "float hidden_layer_%d_weights[] = {\n  ", i);
        if (i == 0) {
            no_of_weights = learner->net->NoOfInputs;
        }
        else {
            no_of_weights = bp_hiddens_in_layer(learner->net, i-1);
        }
        for (j = 0; j < bp_hiddens_in_layer(learner->net, i); j++) {
            for (k = 0; k < no_of_weights; k++) {
                fprintf(fp, "%.10f",
                        learner->net->hiddens[i][j]->weights[k]);
                if (!((j == bp_hiddens_in_layer(learner->net, i)-1) &&
                      (k == no_of_weights-1))) {
                    fprintf(fp, ",");
                }
            }
        }
        fprintf(fp, "%s", "\n};\n\n");
    }

    /* hidden unit biases */
    for (i = 0; i < learner->net->HiddenLayers; i++) {
        fprintf(fp,
                "float hidden_layer_%d_bias[] = {\n  ", i);
        for (j = 0; j < bp_hiddens_in_layer(learner->net, i); j++) {
            fprintf(fp,"%.10f",learner->net->hiddens[i][j]->bias);
            if (j < bp_hiddens_in_layer(learner->net, i)-1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n};\n\n");
    }

    /* output unit weights */
    fprintf(fp, "%s",
            "float output_layer_weights[] = {\n  ");
    for (i = 0; i < learner->net->NoOfOutputs; i++) {
        for (j = 0; j < bp_hiddens_in_layer(learner->net, learner->net->HiddenLayers-1); j++) {
            fprintf(fp, "%.10f",
                    learner->net->outputs[i]->weights[j]);
            if (!((i == learner->net->NoOfOutputs-1) &&
                  (j == bp_hiddens_in_layer(learner->net, learner->net->HiddenLayers-1)-1))) {
                fprintf(fp, ",");
            }
        }
    }
    fprintf(fp, "%s", "\n};\n\n");

    /* output unit biases */
    fprintf(fp, "%s",
            "float output_layer_bias[] = {\n  ");
    for (i = 0; i < learner->net->NoOfOutputs; i++) {
        fprintf(fp, "%.10f",
                learner->net->outputs[i]->bias);
        if (i < learner->net->NoOfOutputs-1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "%s", "\n};\n\n");
    fprintf(fp, "float inputs[%d];\n",learner->net->NoOfInputs);
    fprintf(fp, "float network_inputs[%d];\n",learner->net->NoOfInputs);
    fprintf(fp, "float prev_hiddens[%d];\n",learner->net->NoOfHiddens);
    fprintf(fp, "float hiddens[%d];\n",learner->net->NoOfHiddens);
    fprintf(fp, "float outputs[%d];\n\n",learner->net->NoOfOutputs);

    if (learner->no_of_input_fields > 0) {
        fprintf(fp, "%s", "/* Encode some text into the input units */\n");
        fprintf(fp, "%s", "void encode_text(char * text,\n");
        fprintf(fp, "%s", "                 float * inputs, int no_of_inputs,\n");
        fprintf(fp, "%s", "                 int offset, int max_field_length_chars)\n");

        fprintf(fp, "%s", "{\n  int pos = offset, i, bit, max_chars = strlen(text);\n\n");

        fprintf(fp, "  if (max_chars > (no_of_inputs-offset)/%d) {\n", (int)CHAR_BITS);
        fprintf(fp, "    max_chars = ((no_of_inputs-offset)/%d);\n", (int)CHAR_BITS);
        fprintf(fp, "%s", "  }\n");
        fprintf(fp, "%s", "  if (max_chars > max_field_length_chars) {\n");
        fprintf(fp, "%s", "    max_chars = max_field_length_chars;\n");
        fprintf(fp, "%s", "  }\n\n");

        fprintf(fp, "%s", "  /* for each character in the string */\n");
        fprintf(fp, "%s", "  for (i = 0; i < max_chars; i++) {\n");
        fprintf(fp, "%s", "    /* set the bits for this character */\n");
        fprintf(fp, "    for (bit = 0; bit < %d; bit++, pos++) {\n", (int)CHAR_BITS);
        fprintf(fp, "%s", "      if (text[i] & (1<<bit)) {\n");
        fprintf(fp, "%s", "        inputs[pos] = 0.75f;\n");
        fprintf(fp, "%s", "      }\n");
        fprintf(fp, "%s", "      else {\n");
        fprintf(fp, "%s", "        inputs[pos] = 0.25f;\n");
        fprintf(fp, "%s", "      }\n");
        fprintf(fp, "%s", "    }\n");
        fprintf(fp, "%s", "  }\n");
        fprintf(fp, "%s", "  /* set the remaining inputs within the field to neutral */\n");
        fprintf(fp, "%s", "  while (i < max_field_length_chars) {\n");
        fprintf(fp,       "    for (bit = 0; bit < %d; bit++) {\n", (int)CHAR_BITS);
        fprintf(fp, "%s", "      if (pos >= no_of_inputs) {\n");
        fprintf(fp, "%s", "        i = max_field_length_chars;\n");
        fprintf(fp, "%s", "        break;\n");
        fprintf(fp, "%s", "      }\n");
        fprintf(fp, "%s", "      inputs[pos++] = 0.5f;\n");
        fprintf(fp, "%s", "    }\n");
        fprintf(fp, "%s", "    i++;\n");
        fprintf(fp, "%s", "  }\n");
        fprintf(fp, "%s", "}\n\n");
    }

    fprintf(fp, "%s", "int main(int argc, char* argv[])\n");
    fprintf(fp, "%s", "{\n");

    if (learner->no_of_input_fields == 0) {
        fprintf(fp, "%s", "  int i,j;\n");
    }
    else {
        fprintf(fp, "%s", "  int i,j,pos;\n");
    }
    fprintf(fp, "%s", "  float sum;\n\n");

    if (learner->no_of_input_fields == 0) {
        fprintf(fp, "  if (argc < %d) return -1;\n\n", learner->net->NoOfInputs);
    }
    else {
        fprintf(fp, "  if (argc < %d) return -1;\n\n", learner->no_of_input_fields);
    }

    fprintf(fp, "%s", "  /* Obtain input values from command arguments */\n");
    fprintf(fp, "%s", "  for (i = 1; i < argc; i++) {\n");
    fprintf(fp, "%s", "    if (i > no_of_inputs) return -2;\n");
    fprintf(fp, "%s", "    inputs[i-1] = atof(argv[i]);\n");
    fprintf(fp, "%s", "  }\n\n");

    if (learner->no_of_input_fields == 0) {
        fprintf(fp, "%s", "  /* Normalise inputs into a 0.25 - 0.75 range */\n");
        fprintf(fp, "%s", "  for (i = 0; i < no_of_inputs; i++) {\n");
        fprintf(fp, "%s", "    network_inputs[i] = 0.25f + ((inputs[i] - input_range_min[i])*0.5f/(input_range_max[i] - input_range_min[i]));\n");
        fprintf(fp, "%s", "    if (network_inputs[i] < 0.25f) network_inputs[i] = 0.25f;\n");
        fprintf(fp, "%s", "    if (network_inputs[i] > 0.75f) network_inputs[i] = 0.75f;\n");
        fprintf(fp, "%s", "  }\n\n");
    }
    else {
        fprintf(fp, "%s", "  pos = 0;\n");
        fprintf(fp, "%s", "  for (i = 0; i < no_of_input_fields; i++) {\n");
        fprintf(fp, "%s", "    if (field_length[i] == 0) {\n");
        fprintf(fp, "%s", "      /* Normalise numeric inputs into a 0.25 - 0.75 range */\n");
        fprintf(fp, "%s", "      network_inputs[pos] = 0.25f + ((inputs[i] - input_range_min[i])*0.5f/(input_range_max[i] - input_range_min[i]));\n");
        fprintf(fp, "%s", "      if (network_inputs[pos] < 0.25f) network_inputs[pos] = 0.25f;\n");
        fprintf(fp, "%s", "      if (network_inputs[pos] > 0.75f) network_inputs[pos] = 0.75f;\n");
        fprintf(fp, "%s", "      pos++;\n");
        fprintf(fp, "%s", "    }\n");
        fprintf(fp, "%s", "    else {\n");
        fprintf(fp, "%s", "      /* text value */\n");
        fprintf(fp, "%s", "      encode_text(argv[i+1], network_inputs, no_of_inputs,\n");
        fprintf(fp,       "                  pos, field_length[i]/%d);\n", (int)CHAR_BITS);
        fprintf(fp, "%s", "      pos += field_length[i];\n");
        fprintf(fp, "%s", "    }\n");
        fprintf(fp, "%s", "  }\n\n");
    }

    fprintf(fp, "%s", "  /* Hidden layer 0 */\n");
    fprintf(fp, "%s", "  for (i = 0; i < no_of_hiddens; i++) {\n");
    fprintf(fp, "%s", "    sum = hidden_layer_0_bias[i];\n");
    fprintf(fp, "%s", "    for (j = 0; j < no_of_inputs; j++) {\n");
    fprintf(fp, "%s", "      sum += hidden_layer_0_weights[i*no_of_inputs+j]*network_inputs[j];\n");
    fprintf(fp, "%s", "    }\n");
    fprintf(fp, "%s", "    hiddens[i] = 1.0f / (1.0f + exp(-sum));\n");
    fprintf(fp, "%s", "  }\n");
    fprintf(fp, "%s", "  for (i = 0; i < no_of_hiddens; i++) {\n");
    fprintf(fp, "%s", "    prev_hiddens[i] = hiddens[i];\n");
    fprintf(fp, "%s", "  }\n\n");
    for (i = 1; i < learner->net->HiddenLayers; i++) {
        fprintf(fp, "  /* Hidden layer %d */\n", i);
        fprintf(fp, "  for (i = 0; i < %d; i++) {\n", bp_hiddens_in_layer(learner->net,i));
        fprintf(fp, "    sum = hidden_layer_%d_bias[i];\n",i);
        fprintf(fp, "    for (j = 0; j < %d; j++) {\n",bp_hiddens_in_layer(learner->net,i-1));
        fprintf(fp, "      sum += hidden_layer_%d_weights[i*%d+j]*prev_hiddens[j];\n",i,bp_hiddens_in_layer(learner->net,i-1));
        fprintf(fp, "%s", "    }\n");
        fprintf(fp, "%s", "    hiddens[i] = 1.0f / (1.0f + exp(-sum));\n");
        fprintf(fp, "%s", "  }\n");
        fprintf(fp, "  for (i = 0; i < %d; i++) {\n",bp_hiddens_in_layer(learner->net,i));
        fprintf(fp, "%s", "    prev_hiddens[i] = hiddens[i];\n");
        fprintf(fp, "%s", "  }\n\n");
    }
    fprintf(fp, "%s", "  /* Output layer */\n");
    fprintf(fp, "%s", "  for (i = 0; i < no_of_outputs; i++) {\n");
    fprintf(fp, "%s", "    sum = output_layer_bias[i];\n");
    fprintf(fp, "    for (j = 0; j < %d; j++) {\n",bp_hiddens_in_layer(learner->net,learner->net->HiddenLayers-1));
    fprintf(fp, "      sum += output_layer_weights[i*%d+j]*prev_hiddens[j];\n",bp_hiddens_in_layer(learner->net,learner->net->HiddenLayers-1));
    fprintf(fp, "%s", "    }\n");
    fprintf(fp, "%s", "    outputs[i] = 1.0f / (1.0f + exp(-sum));\n");
    fprintf(fp, "%s", "  }\n\n");

    fprintf(fp, "%s", "  for (i = 0; i < no_of_outputs; i++) {\n");
    fprintf(fp, "%s", "    /* Convert outputs from 0.25 - 0.75 back to their original range */\n");
    fprintf(fp, "%s", "    outputs[i] = output_range_min[i] + ((outputs[i]-0.25f)*(output_range_max[i] - output_range_min[i])/0.5f);\n");
    fprintf(fp, "%s", "    /* Send the outputs to stdout */\n");
    fprintf(fp, "%s", "    printf(\"%.10f\",outputs[i]);\n");
    fprintf(fp, "%s", "    if (i < no_of_outputs-1) {\n");
    fprintf(fp, "%s", "      printf(\" \");\n");
    fprintf(fp, "%s", "    }\n");
    fprintf(fp, "%s", "  }\n\n");
    fprintf(fp, "%s", "  printf(\"\\n\");");
    fprintf(fp, "%s", "\n");
    fprintf(fp, "%s", "  return 0;\n");
    fprintf(fp, "%s", "}\n");
    fclose(fp);
    return 0;
}

/**
 * @brief Exports a trained network as a standalone python class
 * @param learner Deep learner object
 * @param filename The python source file to be produced
 * @returns zero on success
 */
static int deeplearn_export_python(deeplearn * learner, char * filename)
{
    FILE * fp;
    int i, j, k, no_of_weights;

    fp = fopen(filename,"w");
    if (!fp) {
        return -1;
    }

    fprintf(fp, "%s", "#!/usr/bin/python\n\n");
    fprintf(fp, "%s", "import sys\n");
    fprintf(fp, "%s", "import math\n\n");

    fprintf(fp,"%s\n\n","class NeuralNet:");

    if (learner->no_of_input_fields > 0) {
        fprintf(fp, "  no_of_input_fields = %d\n",
                learner->no_of_input_fields);
    }
    fprintf(fp, "  no_of_inputs = %d\n",
            learner->net->NoOfInputs);
    fprintf(fp, "  no_of_hiddens = %d\n",
            learner->net->NoOfHiddens);
    fprintf(fp, "  no_of_outputs = %d\n",
            learner->net->NoOfOutputs);
    fprintf(fp, "  hidden_layers = %d\n\n",
            learner->net->HiddenLayers);

    /* field lengths */
    if (learner->field_length != 0) {
        if (learner->no_of_input_fields > 0) {
            fprintf(fp, "%s", "  field_length = [");
            for (i = 0; i < learner->no_of_input_fields; i++) {
                fprintf(fp, "%d", learner->field_length[i]);
                if (i < learner->no_of_input_fields-1) {
                    fprintf(fp, ",");
                }
            }
            fprintf(fp, "%s", "]\n\n");
        }
    }

    /* ranges */
    fprintf(fp, "%s", "  input_range_min = [");
    for (i = 0; i < learner->net->NoOfInputs; i++) {
        fprintf(fp, "%.10f", learner->input_range_min[i]);
        if (i < learner->net->NoOfInputs-1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "%s", "]\n\n");
    fprintf(fp, "%s", "  input_range_max = [");
    for (i = 0; i < learner->net->NoOfInputs; i++) {
        fprintf(fp, "%.10f", learner->input_range_max[i]);
        if (i < learner->net->NoOfInputs-1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "%s", "]\n\n");
    fprintf(fp, "%s", "  output_range_min = [");
    for (i = 0; i < learner->net->NoOfOutputs; i++) {
        fprintf(fp, "%.10f", learner->output_range_min[i]);
        if (i < learner->net->NoOfOutputs-1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "%s", "]\n\n");
    fprintf(fp, "%s", "  output_range_max = [");
    for (i = 0; i < learner->net->NoOfOutputs; i++) {
        fprintf(fp, "%.10f", learner->output_range_max[i]);
        if (i < learner->net->NoOfOutputs-1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "]\n\n");

    /* hidden unit weights */
    for (i = 0; i < learner->net->HiddenLayers; i++) {
        fprintf(fp,
                "  hidden_layer_%d_weights = [", i);
        if (i == 0) {
            no_of_weights = learner->net->NoOfInputs;
        }
        else {
            no_of_weights = bp_hiddens_in_layer(learner->net, i-1);
        }
        for (j = 0; j < bp_hiddens_in_layer(learner->net, i); j++) {
            for (k = 0; k < no_of_weights; k++) {
                fprintf(fp, "%.10f",
                        learner->net->hiddens[i][j]->weights[k]);
                if (!((j == bp_hiddens_in_layer(learner->net, i)-1) &&
                      (k == no_of_weights-1))) {
                    fprintf(fp, ",");
                }
            }
        }
        fprintf(fp, "]\n\n");
    }

    /* hidden unit biases */
    for (i = 0; i < learner->net->HiddenLayers; i++) {
        fprintf(fp,
                "  hidden_layer_%d_bias = [", i);
        for (j = 0; j < bp_hiddens_in_layer(learner->net, i); j++) {
            fprintf(fp,"%.10f",learner->net->hiddens[i][j]->bias);
            if (j < bp_hiddens_in_layer(learner->net, i)-1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "]\n\n");
    }

    /* output unit weights */
    fprintf(fp,
            "  output_layer_weights = [");
    for (i = 0; i < learner->net->NoOfOutputs; i++) {
        for (j = 0; j < bp_hiddens_in_layer(learner->net, learner->net->HiddenLayers-1); j++) {
            fprintf(fp, "%.10f",
                    learner->net->outputs[i]->weights[j]);
            if (!((i == learner->net->NoOfOutputs-1) &&
                  (j == bp_hiddens_in_layer(learner->net, learner->net->HiddenLayers-1)-1))) {
                fprintf(fp, ",");
            }
        }
    }
    fprintf(fp, "]\n\n");

    /* output unit biases */
    fprintf(fp, "%s",
            "  output_layer_bias = [");
    for (i = 0; i < learner->net->NoOfOutputs; i++) {
        fprintf(fp, "%.10f",
                learner->net->outputs[i]->bias);
        if (i < learner->net->NoOfOutputs-1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "%s", "]\n\n\n");

    if (learner->no_of_input_fields > 0) {
        fprintf(fp, "%s", "  # Encode some text into the input units\n");
        fprintf(fp, "%s", "  def encode_text(this, text,inputs,no_of_inputs,");
        fprintf(fp, "%s", "  offset, max_field_length_chars):\n");

        fprintf(fp, "%s", "    pos = offset\n");
        fprintf(fp, "%s", "    max_chars = len(text)\n\n");

        fprintf(fp, "    if max_chars > (no_of_inputs-offset)/%d:\n", (int)CHAR_BITS);
        fprintf(fp, "      max_chars = ((no_of_inputs-offset)/%d)\n\n", (int)CHAR_BITS);

        fprintf(fp, "%s", "    if max_chars > max_field_length_chars:\n");
        fprintf(fp, "%s", "      max_chars = max_field_length_chars\n\n");

        fprintf(fp, "%s", "    # for each character in the string\n");
        fprintf(fp, "%s", "    for i in range(max_chars):\n");
        fprintf(fp, "%s", "      # set the bits for this character\n");
        fprintf(fp,       "      for bit in range(%d):\n", (int)CHAR_BITS);
        fprintf(fp, "%s", "        if ord(text[i]) & (1 << bit):\n");
        fprintf(fp, "%s", "          inputs.append(0.75)\n");
        fprintf(fp, "%s", "        else:\n");
        fprintf(fp, "%s", "          inputs.append(0.25)\n");
        fprintf(fp, "%s", "        pos = pos + 1\n\n");

        fprintf(fp, "%s", "    # set the remaining inputs within the field to neutral\n");
        fprintf(fp, "%s", "    i = max_chars\n");
        fprintf(fp, "%s", "    while i < max_field_length_chars:\n");
        fprintf(fp,       "      for bit in range (%d):\n", (int)CHAR_BITS);
        fprintf(fp, "%s", "        if pos >= no_of_inputs:\n");
        fprintf(fp, "%s", "          i = max_field_length_chars\n");
        fprintf(fp, "%s", "          break\n");
        fprintf(fp, "%s", "        inputs.append(0.5)\n");
        fprintf(fp, "%s", "        pos = pos + 1\n");
        fprintf(fp, "%s", "      i = i + 1\n\n");
    }

    fprintf(fp, "%s", "  def update(this, inputs):\n");

    if (learner->no_of_input_fields > 0) {
        fprintf(fp, "%s", "    pos = 0\n");
    }
    fprintf(fp, "%s", "    network_inputs = []\n");
    fprintf(fp, "%s", "    prev_hiddens = []\n");
    fprintf(fp, "%s", "    hiddens = []\n");
    fprintf(fp, "%s", "    outputs = []\n\n");
    if (learner->no_of_input_fields == 0) {
        fprintf(fp, "    if len(inputs) < %d:\n", learner->net->NoOfInputs);
    }
    else {
        fprintf(fp, "    if len(inputs) < %d:\n", learner->no_of_input_fields);
    }
    fprintf(fp, "%s", "        return []\n\n");

    if (learner->no_of_input_fields == 0) {
        fprintf(fp, "%s", "    # Normalise inputs into a 0.25 - 0.75 range\n");
        fprintf(fp, "%s", "    for i in range (this.no_of_inputs):\n");
        fprintf(fp, "%s", "      network_inputs.append(0.25 + ((float(inputs[i]) - this.input_range_min[i])*0.5/(this.input_range_max[i] - this.input_range_min[i])))\n");
        fprintf(fp, "%s", "      if network_inputs[i] < 0.25:\n");
        fprintf(fp, "%s", "        network_inputs[i] = 0.25\n");
        fprintf(fp, "%s", "      if network_inputs[i] > 0.75:\n");
        fprintf(fp, "%s", "        network_inputs[i] = 0.75\n");
    }
    else {
        fprintf(fp, "%s", "    for i in range(this.no_of_input_fields):\n");
        fprintf(fp, "%s", "      if this.field_length[i] == 0:\n");
        fprintf(fp, "%s", "        # Normalise numeric inputs into a 0.25 - 0.75 range\n");
        fprintf(fp, "%s", "        network_inputs.append(0.25 + ((float(inputs[i]) - this.input_range_min[i])*0.5/(this.input_range_max[i] - this.input_range_min[i])))\n");
        fprintf(fp, "%s", "        if network_inputs[pos] < 0.25:\n");
        fprintf(fp, "%s", "          network_inputs[pos] = 0.25\n");
        fprintf(fp, "%s", "        if network_inputs[pos] > 0.75:\n");
        fprintf(fp, "%s", "          network_inputs[pos] = 0.75\n\n");
        fprintf(fp, "%s", "        pos = pos + 1\n");
        fprintf(fp, "%s", "      else:\n");
        fprintf(fp, "%s", "        # text value\n");
        fprintf(fp, "%s", "        this.encode_text(inputs[i], network_inputs, this.no_of_inputs,");
        fprintf(fp,       "pos, this.field_length[i]/%d)\n", (int)CHAR_BITS);
        fprintf(fp, "%s", "        pos = pos + this.field_length[i]\n\n");
    }

    fprintf(fp, "%s", "    # Hidden layer 0\n");
    fprintf(fp, "%s", "    for i in range(this.no_of_hiddens):\n");
    fprintf(fp, "%s", "      adder = this.hidden_layer_0_bias[i]\n");
    fprintf(fp, "%s", "      for j in range(this.no_of_inputs):\n");
    fprintf(fp, "%s", "        adder = adder + this.hidden_layer_0_weights[i*this.no_of_inputs+j]*network_inputs[j]\n");
    fprintf(fp, "%s", "      hiddens.append(1.0 / (1.0 + math.exp(-adder)))\n");
    fprintf(fp, "%s", "    for i in range(this.no_of_hiddens):\n");
    fprintf(fp, "%s", "      prev_hiddens.append(hiddens[i])\n\n");
    for (i = 1; i < learner->net->HiddenLayers; i++) {
        fprintf(fp, "    # Hidden layer %d\n", i);
        fprintf(fp, "    for i in range(%d):\n", bp_hiddens_in_layer(learner->net,i));
        fprintf(fp, "      adder = this.hidden_layer_%d_bias[i]\n",i);
        fprintf(fp, "      for j in range(%d):\n",bp_hiddens_in_layer(learner->net,i-1));
        fprintf(fp, "        adder = adder + this.hidden_layer_%d_weights[i*%d+j]*prev_hiddens[j]\n",i,bp_hiddens_in_layer(learner->net,i-1));
        fprintf(fp, "%s", "      hiddens[i] = 1.0 / (1.0 + math.exp(-adder))\n");
        fprintf(fp, "    for i in range(%d):\n",bp_hiddens_in_layer(learner->net,i));
        fprintf(fp, "%s", "      prev_hiddens[i] = hiddens[i]\n\n");
    }
    fprintf(fp, "%s", "    # Output layer\n");
    fprintf(fp, "%s", "    for i in range(this.no_of_outputs):\n");
    fprintf(fp, "%s", "      adder = this.output_layer_bias[i]\n");
    fprintf(fp, "      for j in range(%d):\n",bp_hiddens_in_layer(learner->net,learner->net->HiddenLayers-1));
    fprintf(fp, "        adder = adder + this.output_layer_weights[i*%d+j]*prev_hiddens[j]\n",bp_hiddens_in_layer(learner->net,learner->net->HiddenLayers-1));
    fprintf(fp, "%s", "      outputs.append(1.0 / (1.0 + math.exp(-adder)))\n\n");
    fprintf(fp, "%s", "    # Convert outputs from 0.25 - 0.75 back to their original range\n");
    fprintf(fp, "%s", "    for i in range(this.no_of_outputs):\n");
    fprintf(fp, "%s", "      outputs[i] = this.output_range_min[i] + ((outputs[i]-0.25)*(this.output_range_max[i] - this.output_range_min[i])/0.5)\n\n");

    fprintf(fp, "%s", "    # Return the output unit values as a list\n");
    fprintf(fp, "%s", "    return outputs\n\n\n");

    fprintf(fp, "%s", "# Create an instance of the class\n");
    fprintf(fp, "%s", "net = NeuralNet()\n\n");
    fprintf(fp, "%s", "# Use the commandline arguments as input values\n");
    fprintf(fp, "%s", "print net.update(sys.argv[1:])\n");
    fclose(fp);
    return 0;
}

/**
 * @brief Exports a trained network as a standalone program
 *        file types supported are .c and .py
 * @param learner Deep learner object
 * @param filename The source file to be produced
 * @returns zero on success
 */
int deeplearn_export(deeplearn * learner, char * filename)
{
    int length = strlen(filename);

    if (length > 3) {
        if ((filename[length-3] == '.') &&
            (filename[length-2] == 'p') &&
            (filename[length-1] == 'y')) {
            return deeplearn_export_python(learner, filename);
        }
    }
    return deeplearn_export_c(learner, filename);
}
