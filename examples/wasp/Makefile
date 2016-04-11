
.PHONY: check-syntax

all:
	gcc -Wall -ansi -pedantic -o wasp wasp.c -ldeep -lm -fopenmp

check-syntax:
	gcc -Wall -ansi -pedantic -o wasp wasp.c -ldeep -lm -fopenmp -fsyntax-only

debug:
	gcc -Wall -ansi -pedantic -g -o wasp wasp.c -ldeep -lm -fopenmp

clean:
	rm -f *.o wasp
