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

#include "tests_pooling.h"

static void test_pooling_update()
{
    printf("test_pooling_update...");

    float layer0[20*10*3];
    float layer1[2*2*3];
    int depth = 3;
    int layer0_across = 20;
    int layer0_down = 10;
    int layer1_across = 2;
    int layer1_down = 2;

    /* set some values in the first layer */
    for (int y = 0; y < layer0_down; y++) {
        for (int x = 0; x < layer0_across; x++) {
            for (int d = 0; d < depth; d++) {
                layer0[(y*layer0_across + x)*depth + d] = d+1;
            }
        }
    }

    assert(pooling_update(depth,
                          layer0_across,
                          layer0_down,
                          layer0,
                          layer1_across,
                          layer1_down,
                          layer1)==0);

    for (int y = 0; y < layer1_down; y++) {
        for (int x = 0; x < layer1_across; x++) {
            for (int d = 0; d < depth; d++) {
                float v = layer1[(y*layer1_across + x)*depth + d];
                assert(v > (d+1)-0.01f);
                assert(v < (d+1)+0.01f);
            }
        }
    }

    printf("Ok\n");
}

int run_tests_pooling()
{
    printf("\nRunning pooling tests\n");

    test_pooling_update();

    printf("All pooling tests completed\n");
    return 1;
}
