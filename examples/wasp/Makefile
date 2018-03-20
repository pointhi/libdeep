
.PHONY: check-syntax

all:
	gcc -Wall -ansi -pedantic -mindirect-branch=thunk -o wasp wasp.c -ldeep -lm -fopenmp -ffast-math

check-syntax:
	gcc -Wall -ansi -pedantic -mindirect-branch=thunk -o wasp wasp.c -ldeep -lm -fopenmp -ffast-math -fsyntax-only

debug:
	gcc -Wall -ansi -pedantic -mindirect-branch=thunk -g -o wasp wasp.c -ldeep -lm -fopenmp -ffast-math

clean:
	rm -f *.o wasp
