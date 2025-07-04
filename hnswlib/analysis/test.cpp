#include <cstdio>
#include "extension.h"
#include <climits> // C 风格宏
#include <limits>  // C++ 风格
using namespace std;

void sets(char *data, size_t size) {
    char *data2 = new char[size];
    double t0 = elapsed();
    memcpy(data2, data, size * sizeof(char));
    printf("time for memcpy: %.3f s\n", elapsed() - t0);
}
int main() {
    // C 风格
    // unsigned long min_c = 0;
    // unsigned long max_c = ULONG_MAX;

    // C++ 风格
    unsigned long min_cpp = std::numeric_limits<long>::min();
    unsigned long max_cpp = std::numeric_limits<long>::max();
    std::cout << min_cpp << " " << max_cpp << endl;

    return 0;
}

// p (alg_hnsw0->get_linklist_at_level(0,0))[0]
// 1313, 1784, 4383, 13006, 13452, 15250