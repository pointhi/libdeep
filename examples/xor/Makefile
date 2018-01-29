.PHONY: check-syntax

all:
	gcc -Wall -ansi -pedantic -mindirect-branch=thunk -o xor xor.c -ldeep -lm -fopenmp

check-syntax:
	gcc -Wall -ansi -pedantic -mindirect-branch=thunk -o xor xor.c -ldeep -lm -fopenmp -fsyntax-only

debug:
	gcc -Wall -ansi -pedantic -mindirect-branch=thunk -g -o xor xor.c -ldeep -lm -fopenmp

clean:
	rm -f *.o xor export* *.plist *.png
