CC = g++
CFLAGS_DEBUG = -g
CFLAGS_RELEASE = -O3
CFLAGS_IGNORE = -Wno-write-strings -Wno-unused-result -Wno-format 
# CFLAG_FAISS = -I/usr/local/include -L/usr/local/lib -lfaiss -fopenmp -lopenblas -llapack -ltbb
CFLAG_FAISS = -I/usr/local/include -L/usr/local/lib -fopenmp -lopenblas -llapack -ltbb
SRC = test_merge.cpp
SRC_AVAIL = availability_test.cpp
TARGET = merge
AVAIL = avail

all: release

# alg:
# 	g++ $(CFLAGS_IGNORE) -O3 merge_index.cpp -o algs $(CFLAG_FAISS)

test:
	g++ $(CFLAGS_IGNORE) -O3 test.cpp -o tst $(CFLAG_FAISS)

build:
	g++ $(CFLAGS_IGNORE) -O3 build_index.cpp -o builds $(CFLAG_FAISS)

exp:
	g++ $(CFLAGS_IGNORE) -O3 experiment.cpp -o exps $(CFLAG_FAISS)

debug:
	g++ $(CFLAGS_IGNORE) -g search_index.cpp -o merge $(CFLAG_FAISS)
	gdb ./merge

release:
	g++ $(CFLAGS_IGNORE) -O3 search_index.cpp -o merge $(CFLAG_FAISS)
	./merge 

py:
	python test.py
# clean:
# 	rm -f $(TARGET)
