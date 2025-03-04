CC = g++
CFLAGS_DEBUG = -g
CFLAGS_RELEASE = -O3
CFLAGS_IGNORE = -Wno-write-strings -Wno-unused-result
SRC = test_merge.cpp
SRC_AVAIL = availability_test.cpp
TARGET = merge
AVAIL = avail

all: release

debug:
	g++ -Wno-write-strings -Wno-unused-result -g test_merge.cpp -o merge
	gdb ./merge

release:
	g++ -Wno-write-strings -Wno-unused-result -Wno-format -O3 -g test_merge.cpp -o merge
	./merge 

avail:
	g++ -Wno-write-strings -Wno-unused-result -O3 -g availability_test.cpp -o availability
	./availability 

increase:
	g++ -Wno-write-strings -Wno-unused-result -O3 -g test_merge_multiple_incremental.cpp -o merge-multiple-incremental
	./merge-multiple-incremental

unbias:
	g++ -Wno-write-strings -Wno-unused-result -O3 -g test_merge_unbias.cpp -o merge-unbias
	./merge-unbias

py:
	python test.py
# clean:
# 	rm -f $(TARGET)
