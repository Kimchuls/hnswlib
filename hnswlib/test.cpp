#include <cstdio>
#include "extension.h"
using namespace std;

void sets(char* data, size_t size) {
    char *data2 = new char[size];
    double t0 = elapsed();
    memcpy(data2, data, size * sizeof(char));
    printf("time for memcpy: %.3f s\n", elapsed() - t0);
}
int main() {
    size_t size = 4e9;
    char *data = new char[size];
    double t0 = elapsed();
    memset(data, 0, size * sizeof(char));
    printf("time for memset: %.3f s\n", elapsed() - t0);
    sets(data, size);

    return 0;
}