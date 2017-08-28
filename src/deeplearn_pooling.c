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

#include "deeplearn_pooling.h"

/**
 * @brief Pools the first layer into the second (max pooling)
 * @param depth Depth of the two layers
 * @param unpooled_across Number of units across the first layer
 * @param unpooled_down Number of units down the first layer
 * @param unpooled Array containing the first layer values
 * @param pooled_across Number of units across the second layer
 * @param pooled_down Number of units down the second layer
 * @param pooled Array containing the second layer values
 * @returns zero on success
 */
int pooling_update(int depth,
                   int unpooled_across,
                   int unpooled_down,
                   float unpooled[],
                   int pooled_across,
                   int pooled_down,
                   float pooled[])
{
    /* second layer must be smaller than the first */
    if (pooled_across*pooled_down >
        unpooled_across*unpooled_down)
        return -1;

    /* if layers are the same size then copy the array */
    if (pooled_across*pooled_down ==
        unpooled_across*unpooled_down) {
        memcpy((void*)pooled,(void*)unpooled,
               pooled_across*pooled_down*depth*sizeof(float));
        return 0;
    }

    FLOATCLEAR(pooled, pooled_across*pooled_down*depth);

    /*#pragma omp parallel for*/
    COUNTDOWN(y0, unpooled_down) {
        int y1 = y0 * pooled_down / unpooled_down;
        COUNTDOWN(x0, unpooled_across) {
            int x1 = x0 * pooled_across / unpooled_across;
            int n0 = (y0*unpooled_across + x0)*depth;
            int n1 = (y1*pooled_across + x1)*depth;
            COUNTDOWN(d, depth) {
                if (unpooled[n0+d] > pooled[n1+d])
                    pooled[n1+d] = unpooled[n0+d];
            }
        }
    }

    return 0;
}

/**
 * @brief Unpools the first layer into the second (inverse of max pooling)
 * @param depth Depth of the two layers
 * @param pooling_layer_across Number of units across the first layer
 * @param pooling_layer_down Number of units down the first layer
 * @param pooled Array containing the first layer values
 * @param unpooled_across Number of units across the second layer
 * @param unpooled_down Number of units down the second layer
 * @param unpooled Array containing the second layer values
 * @returns zero on success
 */
int unpooling_update(int depth,
                     int pooled_layer_across,
                     int pooled_layer_down,
                     float pooled_layer[],
                     int unpooled_layer_across,
                     int unpooled_layer_down,
                     float unpooled_layer[])
{
    /* second layer must be smaller than the first */
    if (unpooled_layer_across*unpooled_layer_down >
        pooled_layer_across*pooled_layer_down)
        return -1;

    /* if layers are the same size then copy the array */
    if (unpooled_layer_across*unpooled_layer_down ==
        pooled_layer_across*pooled_layer_down) {
        memcpy((void*)unpooled_layer,(void*)pooled_layer,
               pooled_layer_across*pooled_layer_down*depth*sizeof(float));
        return 0;
    }

    /*#pragma omp parallel for*/
    COUNTDOWN(y_unpooled, unpooled_layer_down) {
        int y_pooled = y_unpooled * pooled_layer_down / unpooled_layer_down;
        COUNTDOWN(x_unpooled, unpooled_layer_across) {
            int x_pooled =
                x_unpooled * pooled_layer_across / unpooled_layer_across;
            int n_pooled =
                (y_pooled*pooled_layer_across + x_pooled)*depth;
            int n_unpooled =
                (y_unpooled*unpooled_layer_across + x_unpooled)*depth;
            COUNTDOWN(d, depth)
                unpooled_layer[n_unpooled+d] = pooled_layer[n_pooled+d];
        }
    }

    return 0;
}
