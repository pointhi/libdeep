
.PHONY: check-syntax

all:
	gcc -Wall -ansi -pedantic -o features features.c -ldeep -lm -fopenmp

check-syntax:
	gcc -Wall -ansi -pedantic -o features features.c -ldeep -lm -fopenmp -fsyntax-only

debug:
	gcc -Wall -ansi -pedantic -g -o features features.c -ldeep -lm -fopenmp

clean:
	rm -f *.o features *.png *.plist
