#include "feijoa.hpp"
#include <cstdio>
#include <ctime>
#include <random>

int main() {
    std::random_device rd;
    std::mt19937_64 generator{rd()};

    clock_t start;

    start = clock();
    for (int i = 0; i < 100000; i++) {
        Feijoa feijoa = Feijoa::random(generator);
        asm volatile("" : : "g"(feijoa));
    }
    fprintf(stderr, "Initialization: %ldk instances/s\n",
            100L * CLOCKS_PER_SEC / (clock() - start));

    start = clock();
    for (int i = 0; i < 100000; i++) {
        auto feijoas = Feijoa::random_many<3>(generator);
        asm volatile("" : : "g"(feijoas));
    }
    fprintf(stderr, "Initialization of 3 Feijoas: 3*%ldk instances/s\n",
            100L * CLOCKS_PER_SEC / (clock() - start));

    Feijoa feijoa = Feijoa::random(generator);
    std::array<char, 4096> page;
    page[0] = 1;
    start = clock();
    for (int i = 0; i < 1000000; i++) {
        uint64_t hash = feijoa.reduce(page.data(), page.size());
        asm volatile("" : : "g"(hash) : "memory");
    }
    fprintf(stderr, "Hashing: %.1f GiB/s\n",
            4096. * 1000000 / 1024 / 1024 / 1024 * CLOCKS_PER_SEC / (clock() - start));

    return 0;
}
