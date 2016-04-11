
.PHONY: check-syntax

all:
	gcc -o webcam webcam.c -lSDL -ldeep -lm -fopenmp

check-syntax:
	gcc -o webcam webcam.c -lSDL -ldeep -lm -fopenmp -fsyntax-only

debug:
	gcc -g -o webcam webcam.c -lSDL -ldeep -lm -fopenmp

clean:
	rm -f *.o webcam
