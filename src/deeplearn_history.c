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

#include "deeplearn_history.h"

/**
 * @brief Initialise a structure containing training history
 * @param history History instance
 * @param filename The image filename to save the history plot
 * @param title Title of the history plot
 */
void deeplearn_history_init(deeplearn_history * history,
                            char filename[], char title[])
{
    history->itterations = 0;
    history->ctr = 0;
    history->index = 0;
    history->step = 1;
    history->interval = 10;

    sprintf(history->filename,"%s", filename);
    sprintf(history->title,"%s", title);
}

/**
 * @brief Update the history of scores during feature learning
 * @param history History instance
 * @param matching score Current score when matching features
 */
void deeplearn_history_update(deeplearn_history * history,
                              float matching_score)
{
    history->itterations++;

    if (history->step == 0) return;

    history->ctr++;
    if (history->ctr >= history->step) {
        if (matching_score == DEEPLEARN_UNKNOWN_ERROR)
            matching_score = 0;

        history->history[history->index] =
            matching_score;
        history->index++;
        history->ctr = 0;

        if (history->index >= DEEPLEARN_HISTORY_SIZE) {
            COUNTUP(i, history->index)
                history->history[i/2] = history->history[i];

            history->index /= 2;
            history->step *= 2;
        }
    }
}

/**
 * @brief Uses gnuplot to plot the training error
 * @param history History instance
 * @param img_width Width of the image in pixels
 * @param img_height Height of the image in pixels
 * @return zero on success
 */
int deeplearn_history_plot(deeplearn_history * history,
                           int img_width, int img_height)
{
    int retval=0;
    FILE * fp;
    char data_filename[256];
    char plot_filename[256];
    char command_str[256];
    float value;
    float max_value = 0.01f;
    char * filename = history->filename;
    char * title = history->title;

    if (strlen(filename) == 0)
        return -1;

    if (strlen(title) == 0)
        return -2;

    sprintf(data_filename,"%s%s",DEEPLEARN_TEMP_DIRECTORY,
            "libdeep_data.dat");
    sprintf(plot_filename,"%s%s",DEEPLEARN_TEMP_DIRECTORY,
            "libdeep_data.plot");

    /* save the data */
    fp = fopen(data_filename,"w");

    if (!fp)
        return -3;

    COUNTUP(index, history->index) {
        value = history->history[index];
        fprintf(fp,"%d    %.10f\n",
                index*history->step,value);
        /* record the maximum error value */
        if (value > max_value)
            max_value = value;
    }
    fclose(fp);

    /* create a plot file */
    fp = fopen(plot_filename,"w");

    if (!fp)
        return -4;

    fprintf(fp,"%s","reset\n");
    fprintf(fp,"set title \"%s\"\n",title);
    fprintf(fp,"set xrange [0:%d]\n",
            history->index*history->step);
    fprintf(fp,"set yrange [0:%f]\n",max_value*102/100);
    fprintf(fp,"%s","set lmargin 9\n");
    fprintf(fp,"%s","set rmargin 2\n");
    fprintf(fp,"%s","set xlabel \"Time Step\"\n");
    fprintf(fp,"%s","set ylabel \"Training Error\"\n");

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
