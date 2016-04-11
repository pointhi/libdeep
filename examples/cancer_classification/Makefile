
.PHONY: check-syntax

check-syntax:
	gcc -Wall -ansi -pedantic -o cancerdetect cancerdetect.c -ldeep -lm -fopenmp -fsyntax-only

all:
	gcc -Wall -ansi -pedantic -o cancerdetect cancerdetect.c -ldeep -lm -fopenmp

debug:
	gcc -Wall -ansi -pedantic -g -o cancerdetect cancerdetect.c -ldeep -lm -fopenmp

clean:
	rm -f *.o cancerdetect
