APP=tests

.PHONY: check-syntax

all:
	gcc -Wall -std=c99 -pedantic -g -o $(APP) *.c ../src/*.c -I../src -lm
check-syntax:
	gcc -Wall -std=c99 -pedantic -g -o $(APP) *.c ../src/*.c -I../src -lm -fsyntax-only
debug:
	gcc -Wall -std=c99 -pedantic -g -o $(APP) *.c ../src/*.c -I../src -lm
clean:
	rm -f ${APP}
