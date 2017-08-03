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

#ifndef DEEPLEARN_GLOBALS_H
#define DEEPLEARN_GLOBALS_H

#define DEEPLEARN_TEMP_DIRECTORY          "/tmp/"
#define DEEPLEARN_HISTORY_SIZE            1024
#define DEEPLEARN_UNKNOWN_ERROR           9999
#define DEEPLEARN_UNKNOWN_VALUE          -9999
#define DEEPLEARN_MAX_FIELD_LENGTH_CHARS  1024
#define DEEPLEARN_MAX_CSV_INPUTS          2048
#define DEEPLEARN_MAX_CSV_OUTPUTS         1024

/* The number of bits per character in a text string */
#define CHAR_BITS               (sizeof(char)*8)

#define AUTOCODER_UNKNOWN      -9999
#define AUTOCODER_DROPPED_OUT  -9999

#define AF_SIGMOID              0
#define AF_TANH                 1
#define AF_LINEAR               2

#define ACTIVATION_FUNCTION     AF_SIGMOID

#if ACTIVATION_FUNCTION == AF_SIGMOID
#define AF(adder) (1.0f / (1.0f + exp(-(adder))))
#elif ACTIVATION_FUNCTION == AF_TANH
#define AF(adder) ((((2.0f / (1.0f + exp(-(2*adder)))) - 1.0f)*0.5f)+0.5f)
#elif ACTIVATION_FUNCTION == AF_LINEAR
#define AF(adder) ((adder) < 1.0f ? ((adder) > -1.0f ? (((adder)*0.5f)+0.5f) : 0.0f) : 1.0f)
#endif

#define PIXEL_TO_FLOAT(p)       (0.25f + ((p)/(2*255.0f)))

#define FOR(i, start, end) for (int (i) = (start); (i) < (end); (i)++)
#define COUNTDOWN(i, end) for (int (i) = (end-1); (i) >= 0; (i)--)

#endif
