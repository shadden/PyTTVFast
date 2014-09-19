CC=gcc
CFLAGS= -fPIC -O3 -lm

SOURCES=TTVFast.c  #machine-epsilon.c kepcart2.c ttv-map-jacobi.c
OBJECTS=$(SOURCES:.c=.o)


all: $(OBJECTS)
	$(CC) $(CFLAGS) -shared -o libttvfast.so $^ 
run_TTVFast: run_TTVFast.c TTVFast.o
	$(CC) $(CFLAGS) -o $@ $^
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	rm -rf *.o
