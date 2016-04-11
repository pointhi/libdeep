
.PHONY: check-syntax

check-syntax:
	gcc -o webcam webcam.c -lSDL -ldeep -lm -fopenmp -fsyntax-only

all:
	gcc -o webcam webcam.c -lSDL -ldeep -lm -fopenmp

debug:
	gcc -g -o webcam webcam.c -lSDL -ldeep -lm -fopenmp

clean:
	rm -f *.o webcam
