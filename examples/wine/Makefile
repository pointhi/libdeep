.PHONY: check-syntax

check-syntax:
	gcc -Wall -ansi -pedantic -o winequality winequality.c -ldeep -lm -fopenmp -fsyntax-only

all:
	gcc -Wall -ansi -pedantic -o winequality winequality.c -ldeep -lm -fopenmp

debug:
	gcc -Wall -ansi -pedantic -g -o winequality winequality.c -ldeep -lm -fopenmp

clean:
	rm -f *.o winequality
