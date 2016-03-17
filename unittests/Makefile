APP=tests

all:
	gcc -Wall -std=c99 -pedantic -g -o $(APP) *.c ../src/*.c -I../src -lm
debug:
	gcc -Wall -std=c99 -pedantic -g -o $(APP) *.c ../src/*.c -I../src -lm
clean:
	rm -f ${APP}
