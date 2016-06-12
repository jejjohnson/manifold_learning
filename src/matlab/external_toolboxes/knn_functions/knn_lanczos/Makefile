CC    = g++

LIB   = -lgsl -lgslcblas -lm
MACRO = -DHAVE_INLINE
OPTM  = -O2
FLAG  = $(LIB) $(MACRO) $(OPTM)

PROG  = knn

## Compile
all: $(PROG)

$(PROG): main.o knn_algo.o linear_algebra.o dynamic_array.o aux_func.o
	$(CC) main.o knn_algo.o linear_algebra.o dynamic_array.o aux_func.o -o $(PROG) $(FLAG)

main.o: main.cpp knn_algo.h aux_func.h
	$(CC) -c main.cpp $(OPTM)

knn_algo.o: knn_algo.cpp knn_algo.h linear_algebra.h aux_func.h
	$(CC) -c knn_algo.cpp $(OPTM)

linear_algebra.o: linear_algebra.cpp linear_algebra.h aux_func.h
	$(CC) -c linear_algebra.cpp $(OPTM)

dynamic_array.o: dynamic_array.cpp dynamic_array.h
	$(CC) -c dynamic_array.cpp $(OPTM)

aux_func.o: aux_func.cpp aux_func.h
	$(CC) -c aux_func.cpp $(OPTM)

## Clean
clean:
	rm -f *.o
