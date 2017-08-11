APP=tests

.PHONY: check-syntax

all:
	gcc -Wall -std=c99 -pedantic -g -o $(APP) *.c ../src/*.c -I../src -lm -fopenmp
check-syntax:
	gcc -Wall -std=c99 -pedantic -fsanitize=address -g -o $(APP) *.c ../src/*.c -I../src -lm -fsyntax-only
debug:
	gcc -Wall -std=c99 -pedantic -g -o $(APP) *.c ../src/*.c -I../src -lm -fopenmp
debugstack:
	gcc -Wall -std=c99 -pedantic -fsanitize=address -g -o $(APP) *.c ../src/*.c -I../src -lm -fopenmp
clean:
	rm -f ${APP} *.plist
