APP=tests

all:
	rm -f training.png
	gcc -Wall -std=c99 -pedantic -g -o $(APP) *.c ../src/*.c -I../src -lm
debug:
	rm -f training.png
	gcc -Wall -std=c99 -pedantic -g -o $(APP) *.c ../src/*.c -I../src -lm
clean:
	rm -f training.png
	rm -f ${APP}
