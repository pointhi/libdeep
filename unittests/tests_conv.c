/*
  Copyright (C) 2015  Bob Mottram <bob@robotics.uk.to>

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

#include "tests_conv.h"

static void test_conv_init()
{
    printf("test_conv_init...");

    int no_of_layers = 3;
    int image_width = 256;
    int image_height = 256;
    int image_depth = 3;
    int no_of_features = 10*10;
    int feature_width = 8;
    int final_image_width = 64;
    int final_image_height = 64;
    float match_threshold[] = { 0.0f, 0.0f, 0.0f };
    deeplearn_conv conv;

    assert(conv_init(no_of_layers,
                     image_width, image_height, image_depth,
                     no_of_features, feature_width,
                     final_image_width, final_image_height,
                     &match_threshold[0], &conv) == 0);
    conv_free(&conv);

    printf("Ok\n");
}


int run_tests_conv()
{
    printf("\nRunning convolution tests\n");

    test_conv_init();

    printf("All convolution tests completed\n");
    return 1;
}
