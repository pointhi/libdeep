APP=tests

.PHONY: check-syntax

check-syntax:
	gcc -Wall -std=c99 -pedantic -g -o $(APP) *.c ../src/*.c -I../src -lm -fsyntax-only
all:
	gcc -Wall -std=c99 -pedantic -g -o $(APP) *.c ../src/*.c -I../src -lm
debug:
	gcc -Wall -std=c99 -pedantic -g -o $(APP) *.c ../src/*.c -I../src -lm
clean:
	rm -f ${APP}
