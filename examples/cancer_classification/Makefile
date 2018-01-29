all:
	gcc -Wall -ansi -pedantic -mindirect-branch=thunk -o cancerdetect cancerdetect.c -ldeep -lm -fopenmp

debug:
	gcc -Wall -ansi -pedantic -mindirect-branch=thunk -g -o cancerdetect cancerdetect.c -ldeep -lm -fopenmp

clean:
	rm -f *.o cancerdetect export* *.plist
