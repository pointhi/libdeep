.PHONY: check-syntax

all:
	gcc -Wall -ansi -pedantic -o xor xor.c -ldeep -lm -fopenmp

check-syntax:
	gcc -Wall -ansi -pedantic -o xor xor.c -ldeep -lm -fopenmp -fsyntax-only

debug:
	gcc -Wall -ansi -pedantic -g -o xor xor.c -ldeep -lm -fopenmp

clean:
	rm -f *.o xor export* *.plist *.png
