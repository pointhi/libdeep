/*
 Copyright (C) 2016  Bob Mottram <bob@freedombone.net>

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

#include "tests_dnc.h"

static void test_dnc_init() {
    deeplearn_dnc learner;
    int memory_size = 100;
    int memory_width = 8;
    int no_of_inputs = 4;
    int no_of_hiddens = 5;
    int hidden_layers = 3;
    int no_of_outputs = 3;
    float error_threshold[] = {0.1f, 0.1f, 0.1f, 0.1f};
    unsigned int random_seed = 3672;

    int retval =
        deeplearn_dnc_init(&learner,
                           memory_size, memory_width,
                           no_of_inputs,
                           no_of_hiddens,
                           hidden_layers,
                           no_of_outputs,
                           error_threshold,
                           &random_seed);
    assert(retval == 0);
    deeplearn_dnc_free(&learner);
}

int run_tests_dnc()
{
    printf("\nRunning dnc tests\n");

    test_dnc_init();

    printf("All dnc tests completed\n");
    return 1;
}
