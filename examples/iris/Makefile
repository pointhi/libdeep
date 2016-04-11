
.PHONY: check-syntax

all:
	gcc -Wall -ansi -pedantic -o irisclassifier iris.c -ldeep -lm -fopenmp

check-syntax:
	gcc -Wall -ansi -pedantic -o irisclassifier iris.c -ldeep -lm -fopenmp -fsyntax-only

debug:
	gcc -Wall -ansi -pedantic -g -o irisclassifier iris.c -ldeep -lm -fopenmp

clean:
	rm -f *.o irisclassifier
