#define FEIJOA_TESTS
#include "feijoa.hpp"
#include <cassert>

bool eq(Feijoa::Vector a, Feijoa::Vector b) { return a.low() == b.low() && a.high() == b.high(); }

#ifdef __x86_64__
#define ASSERT_IF_PDEP(...)                                                                        \
    do {                                                                                           \
        if (Feijoa::RepeatedlySquarablePdep::is_available()) {                                     \
            assert(__VA_ARGS__);                                                                   \
        }                                                                                          \
    } while (false)
#else
#define ASSERT_IF_PDEP(...)                                                                        \
    do {                                                                                           \
    } while (false)
#endif

int main() {
    char buffer[] =
        "\x66\xcb\xe4\x6c\x1c\x17\xf6\xb3\x75\x20\x79\x09\x0c\xa9\x0f\xed\xcd\x18\xc4\xc1\x82\xc3"
        "\x1e\xb8\x0e\xf6\x94\x0c\x69\x9a\xfa\x08\x9a\x1c\x30\x96\x41\x11\x8f\xb1\x2e\xc3\x66\x44"
        "\x43\xda\xc0\xb5\x8d\xe1\x29\xc1\xf5\x2c\xfa\x8e\x40\xfa\x05\x38\x2a\x59\x25\x1a";

    {
        // x^64
        Feijoa feijoa(0);
        assert(feijoa.low_p_low_x_128_div_p.low() == 0);
        assert(feijoa.low_p_low_x_128_div_p.high() == 0);
        assert(feijoa.x_128_x_192.low() == 0);
        assert(feijoa.x_128_x_192.high() == 0);
        assert(feijoa.x_512_x_576.low() == 0);
        assert(feijoa.x_512_x_576.high() == 0);
        assert(feijoa.get_seed() == 0);
        assert(feijoa == feijoa);
        assert(feijoa != Feijoa{1});
        // (x^2 + 1)^2 = x^4 + 1
        assert(eq(feijoa.square(Feijoa::Vector{0b101, 0}), Feijoa::Vector{0b10001, 0}));
        // (x^32)^2 = x^64
        assert(eq(feijoa.square(Feijoa::Vector{uint64_t{1} << 32, 0}), Feijoa::Vector{0, 1}));
        // (x^64)^2 = 0
        assert(eq(feijoa.square(Feijoa::Vector{0, 1}), Feijoa::Vector{0, 0}));
        // (x^33 + x^32 + x^2 + 1)^2 mod x^64 = x^4 + 1
        assert((Feijoa::RepeatedlySquarableClmul{feijoa, 0x300000005}.squared().reduced() == 0x11));
        ASSERT_IF_PDEP(
            (Feijoa::RepeatedlySquarablePdep{feijoa, 0x300000005}.squared().reduced() == 0x11));
        assert(eq(feijoa.shift_128(Feijoa::Vector{123, 456}), Feijoa::Vector{0, 0}));
        assert(eq(feijoa.shift_512(Feijoa::Vector{123, 456}), Feijoa::Vector{0, 0}));
        assert(feijoa.reduce(Feijoa::Vector{123, 456}) == 123);
        assert(feijoa.reduce(buffer, 64) == 0x8efa2cf5c129e18d);
        assert(feijoa.reduce(buffer + 16, 48) == 0x8efa2cf5c129e18d);
        assert(feijoa.reduce(buffer, 0) == 0);
        assert(!feijoa.is_irreducible<Feijoa::RepeatedlySquarableClmul>());
        assert(!feijoa.is_quasi_irreducible<Feijoa::RepeatedlySquarableClmul>().first);
        ASSERT_IF_PDEP(!feijoa.is_irreducible<Feijoa::RepeatedlySquarablePdep>());
        ASSERT_IF_PDEP(!feijoa.is_quasi_irreducible<Feijoa::RepeatedlySquarablePdep>().first);
    }

    {
        // x^64 + x^63 + x^62 + x^61 + x^58 + x^54 + x^48 + x^46 + x^43 + x^41 + x^40 + x^39 + x^37
        //     + x^36 + x^35 + x^33 + x^30 + x^28 + x^27 + x^26 + x^25 + x^23 + x^22 + x^21 + x^20
        //     + x^19 + x^8 + x^7 + x^6 + x^2 + 1
        Feijoa feijoa(0xe4414bba5ef801c5);
        assert(feijoa.low_p_low_x_128_div_p.low() == 0xe4414bba5ef801c5U);
        assert(feijoa.low_p_low_x_128_div_p.high() == 0x9cd26aeea99afeb4U);
        assert(feijoa.x_128_x_192.low() == 0xe62e245859af4764U);
        assert(feijoa.x_128_x_192.high() == 0x329ed7d43d59826cU);
        assert(feijoa.x_512_x_576.low() == 0xd0b65ca1f87a3466U);
        assert(feijoa.x_512_x_576.high() == 0x03cbfc7a304ea1dcU);
        assert(feijoa.get_seed() == 0xe4414bba5ef801c5);
        assert(feijoa == feijoa);
        assert(feijoa != Feijoa{1});
        Feijoa::Vector a{0x7806f7d4cc65b145, 0xce061518d88c8a77};
        assert(feijoa.reduce(a) == 0xb2a3023836224dae);
        assert(feijoa.reduce(feijoa.square(a)) == 0x47ff962ca9e606df);
        assert(feijoa.reduce(feijoa.shift_128(a)) == 0x6f010a6522f6d04a);
        assert(feijoa.reduce(feijoa.shift_512(a)) == 0x3c04fde0b1149209);
        assert((Feijoa::RepeatedlySquarableClmul{feijoa, 0xb2a3023836224dae}.squared().reduced() ==
                0x47ff962ca9e606df));
        ASSERT_IF_PDEP(
            (Feijoa::RepeatedlySquarablePdep{feijoa, 0xb2a3023836224dae}.squared().reduced() ==
             0x47ff962ca9e606df));
        assert(!feijoa.is_irreducible<Feijoa::RepeatedlySquarableClmul>());
        assert(!feijoa.is_quasi_irreducible<Feijoa::RepeatedlySquarableClmul>().first);
        ASSERT_IF_PDEP(!feijoa.is_irreducible<Feijoa::RepeatedlySquarablePdep>());
        ASSERT_IF_PDEP(!feijoa.is_quasi_irreducible<Feijoa::RepeatedlySquarablePdep>().first);
        assert(feijoa.reduce(buffer, 64) == 0x45d383006154c9b7);
        assert(feijoa.reduce(buffer + 16, 48) == 0xfe98ad376a5e9a75);
        assert(feijoa.reduce(buffer, 0) == 0xe4414bba5ef801c5);
    }

    {
        // x^64 + x^62 + x^61 + x^60 + x^57 + x^56 + x^54 + x^53 + x^48 + x^47 + x^46 + x^45 +
        // x^41
        //     + x^37 + x^35 + x^34 + x^32 + x^31 + x^30 + x^29 + x^28 + x^27 + x^25 + x^19 +
        //     x^18
        //     + x^17 + x^8 + x^4 + x^3 + x^2 + 1
        Feijoa feijoa{0x7361e22dfa0e011d};
        assert(feijoa.low_p_low_x_128_div_p.low() == 0x7361e22dfa0e011dU);
        assert(feijoa.low_p_low_x_128_div_p.high() == 0x619b551519cfaa41U);
        assert(feijoa.x_128_x_192.low() == 0x9822de4a3652b45dU);
        assert(feijoa.x_128_x_192.high() == 0xd937bd8ad5c7974dU);
        assert(feijoa.x_512_x_576.low() == 0x7585192ed81b3087U);
        assert(feijoa.x_512_x_576.high() == 0xee9e707757f4e581U);
        assert(feijoa.get_seed() == 0x7361e22dfa0e011d);
        assert(feijoa == feijoa);
        assert(feijoa != Feijoa{1});
        Feijoa::Vector a{0xeebe4ebfdbf8869d, 0xc17b85927574fab9};
        assert(feijoa.reduce(a) == 0x412f5c88a5785e87);
        assert(feijoa.reduce(feijoa.square(a)) == 0x9682819ce543af15);
        assert(feijoa.reduce(feijoa.shift_128(a)) == 0x61017f24d3cfdfec);
        assert(feijoa.reduce(feijoa.shift_512(a)) == 0xd4d2d6bc13a5c724);
        assert((Feijoa::RepeatedlySquarableClmul{feijoa, 0x412f5c88a5785e87}.squared().reduced() ==
                0x9682819ce543af15));
        ASSERT_IF_PDEP(
            (Feijoa::RepeatedlySquarablePdep{feijoa, 0x412f5c88a5785e87}.squared().reduced() ==
             0x9682819ce543af15));
        assert(feijoa.is_irreducible<Feijoa::RepeatedlySquarableClmul>());
        assert(feijoa.is_quasi_irreducible<Feijoa::RepeatedlySquarableClmul>().first);
        ASSERT_IF_PDEP(feijoa.is_irreducible<Feijoa::RepeatedlySquarablePdep>());
        ASSERT_IF_PDEP(feijoa.is_quasi_irreducible<Feijoa::RepeatedlySquarablePdep>().first);
        assert(feijoa.reduce(buffer, 64) == 0xb2d8e830efef6b49);
        assert(feijoa.reduce(buffer + 16, 48) == 0x6cc9738b07e91f18);
        assert(feijoa.reduce(buffer, 0) == 0x7361e22dfa0e011d);
    }

    {
        // (
        //     x^32 + x^29 + x^28 + x^26 + x^25 + x^23 + x^22 + x^21 + x^18 + x^14 + x^13 + x^12
        //     + x^10 + x^9 + x^8 + x^6 + x^5 + x^4 + x^3 + x^2 + 1)
        // ) * (
        //     x^32 + x^30 + x^28 + x^27 + x^25 + x^23 + x^22 + x^21 + x^20 + x^17 + x^16 + x^15
        //     + x^14 + x^13 + x^12 + x^10 + x^9 + x^8 + x^7 + x^4 + x^3 + x + 1
        // ) = x^64 + x^62 + x^61 + x^57 + x^56 + x^51 + x^48 + x^37 + x^36 + x^35 + x^30 + x^23
        //     + x^22 + x^21 + x^20 + x^19 + x^16 + x^13 + x^11 + x^5 + x^4 + x^3 + x^2 + x + 1
        Feijoa feijoa{0x6309003840f9283f};
        assert(!feijoa.is_irreducible<Feijoa::RepeatedlySquarableClmul>());
        assert(feijoa.is_quasi_irreducible<Feijoa::RepeatedlySquarableClmul>().first);
        ASSERT_IF_PDEP(!feijoa.is_irreducible<Feijoa::RepeatedlySquarablePdep>());
        ASSERT_IF_PDEP(feijoa.is_quasi_irreducible<Feijoa::RepeatedlySquarablePdep>().first);
    }

    std::mt19937_64 generator{0};
    assert(Feijoa::random<Feijoa::RepeatedlySquarableClmul>(generator)
               .is_irreducible<Feijoa::RepeatedlySquarableClmul>());
    ASSERT_IF_PDEP(Feijoa::random<Feijoa::RepeatedlySquarablePdep>(generator)
                       .is_irreducible<Feijoa::RepeatedlySquarablePdep>());

    auto feijoas = Feijoa::random_many<100>(generator);
    for (Feijoa &fejjoa : feijoas) {
        assert(fejjoa.is_irreducible<Feijoa::RepeatedlySquarableClmul>());
    }
    std::sort(feijoas.begin(), feijoas.end(),
              [](Feijoa &a, Feijoa &b) { return a.get_seed() < b.get_seed(); });
    for (size_t i = 1; i < feijoas.size(); i++) {
        assert(feijoas[i - 1] != feijoas[i]);
    }

    return 0;
}
