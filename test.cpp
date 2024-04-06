#define FEIJOA_TESTS
#include "feijoa.hpp"
#include <cassert>

bool eq(FEIJOA_VECTOR_TYPE a, FEIJOA_VECTOR_TYPE b) { return a[0] == b[0] && a[1] == b[1]; }

int main() {
    for (uint64_t a : std::array<uint64_t, 4>{0xfcf980c83018a6d2, 0x3856dbeaf383ce21,
                                              0xf4d0134ced3abf0e, 0x471608aa92303b90}) {
        uint64_t fixup = Feijoa::fixup_coprime(a);
        assert((fixup >> 2) == (a >> 2));
        assert(fixup & 1);
        assert(!__builtin_parityl(fixup));
    }

    char buffer[] =
        "\x66\xcb\xe4\x6c\x1c\x17\xf6\xb3\x75\x20\x79\x09\x0c\xa9\x0f\xed\xcd\x18\xc4\xc1\x82\xc3"
        "\x1e\xb8\x0e\xf6\x94\x0c\x69\x9a\xfa\x08\x9a\x1c\x30\x96\x41\x11\x8f\xb1\x2e\xc3\x66\x44"
        "\x43\xda\xc0\xb5\x8d\xe1\x29\xc1\xf5\x2c\xfa\x8e\x40\xfa\x05\x38\x2a\x59\x25\x1a";

    {
        // x^64
        Feijoa feijoa(0);
        assert(feijoa.low_p_low_x_128_div_p[0] == 0);
        assert(feijoa.low_p_low_x_128_div_p[1] == 0);
        assert(feijoa.x_128_x_192[0] == 0);
        assert(feijoa.x_128_x_192[1] == 0);
        assert(feijoa.x_512_x_576[0] == 0);
        assert(feijoa.x_512_x_576[1] == 0);
        assert(feijoa.get_seed() == 0);
        assert(feijoa == feijoa);
        assert(feijoa != Feijoa{1});
        // (x^2 + 1)^2 = x^4 + 1
        assert(eq(feijoa.square(Feijoa::polynomial_pair(0b101, 0)),
                  Feijoa::polynomial_pair(0b10001, 0)));
        // (x^32)^2 = x^64
        assert(eq(feijoa.square(Feijoa::polynomial_pair(uint64_t{1} << 32, 0)),
                  Feijoa::polynomial_pair(0, 1)));
        // (x^64)^2 = 0
        assert(eq(feijoa.square(Feijoa::polynomial_pair(0, 1)), Feijoa::polynomial_pair(0, 0)));
        if (Feijoa::has_pdep()) {
            // (x^33 + x^32 + x^2 + 1)^2 = x^66 + x^64 + x^4 + 1
            assert(eq(feijoa.square(0x300000005, std::true_type{}),
                      Feijoa::polynomial_pair(0b10001, 0b101)));
        }
        assert(
            eq(feijoa.shift_128(Feijoa::polynomial_pair(123, 456)), Feijoa::polynomial_pair(0, 0)));
        assert(
            eq(feijoa.shift_512(Feijoa::polynomial_pair(123, 456)), Feijoa::polynomial_pair(0, 0)));
        assert(feijoa.reduce(Feijoa::polynomial_pair(123, 456)) == 123);
        assert(feijoa.reduce(buffer, 64) == 0x8efa2cf5c129e18d);
        assert(feijoa.reduce(buffer + 16, 48) == 0x8efa2cf5c129e18d);
        assert(feijoa.reduce(buffer, 0) == 0);
        assert(!feijoa.is_irreducible(std::false_type{}));
        assert(!feijoa.is_quasi_irreducible(std::false_type{}).first);
        if (Feijoa::has_pdep()) {
            assert(!feijoa.is_irreducible(std::true_type{}));
            assert(!feijoa.is_quasi_irreducible(std::true_type{}).first);
        }
    }

    {
        // x^64 + x^63 + x^62 + x^61 + x^58 + x^54 + x^48 + x^46 + x^43 + x^41 + x^40 + x^39 + x^37
        //     + x^36 + x^35 + x^33 + x^30 + x^28 + x^27 + x^26 + x^25 + x^23 + x^22 + x^21 + x^20
        //     + x^19 + x^8 + x^7 + x^6 + x^2 + 1
        Feijoa feijoa(0xe4414bba5ef801c5);
        assert((uint64_t)feijoa.low_p_low_x_128_div_p[0] == 0xe4414bba5ef801c5U);
        assert((uint64_t)feijoa.low_p_low_x_128_div_p[1] == 0x9cd26aeea99afeb4U);
        assert((uint64_t)feijoa.x_128_x_192[0] == 0xe62e245859af4764U);
        assert((uint64_t)feijoa.x_128_x_192[1] == 0x329ed7d43d59826cU);
        assert((uint64_t)feijoa.x_512_x_576[0] == 0xd0b65ca1f87a3466U);
        assert((uint64_t)feijoa.x_512_x_576[1] == 0x03cbfc7a304ea1dcU);
        assert(feijoa.get_seed() == 0xe4414bba5ef801c5);
        assert(feijoa == feijoa);
        assert(feijoa != Feijoa{1});
        auto a = Feijoa::polynomial_pair(0x7806f7d4cc65b145, 0xce061518d88c8a77);
        assert(feijoa.reduce(a) == 0xb2a3023836224dae);
        assert(feijoa.reduce(feijoa.square(a)) == 0x47ff962ca9e606df);
        assert(feijoa.reduce(feijoa.shift_128(a)) == 0x6f010a6522f6d04a);
        assert(feijoa.reduce(feijoa.shift_512(a)) == 0x3c04fde0b1149209);
        if (Feijoa::has_pdep()) {
            assert(eq(feijoa.square(0xe30d03531263e5f5, std::true_type{}),
                      Feijoa::polynomial_pair(0x0104140554115511, 0x5405005100051105)));
        }
        assert(!feijoa.is_irreducible(std::false_type{}));
        assert(!feijoa.is_quasi_irreducible(std::false_type{}).first);
        if (Feijoa::has_pdep()) {
            assert(!feijoa.is_irreducible(std::true_type{}));
            assert(!feijoa.is_quasi_irreducible(std::true_type{}).first);
        }
        assert(feijoa.reduce(buffer, 64) == 0x45d383006154c9b7);
        assert(feijoa.reduce(buffer + 16, 48) == 0xfe98ad376a5e9a75);
        assert(feijoa.reduce(buffer, 0) == 0xe4414bba5ef801c5);
    }

    {
        // x^64 + x^62 + x^61 + x^60 + x^57 + x^56 + x^54 + x^53 + x^48 + x^47 + x^46 + x^45 + x^41
        //     + x^37 + x^35 + x^34 + x^32 + x^31 + x^30 + x^29 + x^28 + x^27 + x^25 + x^19 + x^18
        //     + x^17 + x^8 + x^4 + x^3 + x^2 + 1
        Feijoa feijoa{0x7361e22dfa0e011d};
        assert((uint64_t)feijoa.low_p_low_x_128_div_p[0] == 0x7361e22dfa0e011dU);
        assert((uint64_t)feijoa.low_p_low_x_128_div_p[1] == 0x619b551519cfaa41U);
        assert((uint64_t)feijoa.x_128_x_192[0] == 0x9822de4a3652b45dU);
        assert((uint64_t)feijoa.x_128_x_192[1] == 0xd937bd8ad5c7974dU);
        assert((uint64_t)feijoa.x_512_x_576[0] == 0x7585192ed81b3087U);
        assert((uint64_t)feijoa.x_512_x_576[1] == 0xee9e707757f4e581U);
        assert(feijoa.get_seed() == 0x7361e22dfa0e011d);
        assert(feijoa == feijoa);
        assert(feijoa != Feijoa{1});
        auto a = Feijoa::polynomial_pair(0xeebe4ebfdbf8869d, 0xc17b85927574fab9);
        assert(feijoa.reduce(a) == 0x412f5c88a5785e87);
        assert(feijoa.reduce(feijoa.square(a)) == 0x9682819ce543af15);
        assert(feijoa.reduce(feijoa.shift_128(a)) == 0x61017f24d3cfdfec);
        assert(feijoa.reduce(feijoa.shift_512(a)) == 0xd4d2d6bc13a5c724);
        if (Feijoa::has_pdep()) {
            assert(eq(feijoa.square(0xc78c0896c394c3cf, std::true_type{}),
                      Feijoa::polynomial_pair(0x5005411050055055, 0x5015405000404114)));
        }
        assert(feijoa.is_irreducible(std::false_type{}));
        assert(feijoa.is_quasi_irreducible(std::false_type{}).first);
        if (Feijoa::has_pdep()) {
            assert(feijoa.is_irreducible(std::true_type{}));
            assert(feijoa.is_quasi_irreducible(std::true_type{}).first);
        }
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
        assert(!feijoa.is_irreducible(std::false_type{}));
        assert(feijoa.is_quasi_irreducible(std::false_type{}).first);
        if (Feijoa::has_pdep()) {
            assert(!feijoa.is_irreducible(std::true_type{}));
            assert(feijoa.is_quasi_irreducible(std::true_type{}).first);
        }
    }

    std::mt19937_64 generator{0};
    assert(Feijoa::random(generator, std::false_type{}).is_irreducible());
    if (Feijoa::has_pdep()) {
        assert(Feijoa::random(generator, std::true_type{}).is_irreducible());
    }

    auto feijoas = Feijoa::random_many<100>(generator);
    for (Feijoa &fejjoa : feijoas) {
        assert(fejjoa.is_irreducible());
    }
    std::sort(feijoas.begin(), feijoas.end(),
              [](Feijoa &a, Feijoa &b) { return a.get_seed() < b.get_seed(); });
    for (size_t i = 1; i < feijoas.size(); i++) {
        assert(feijoas[i - 1] != feijoas[i]);
    }

    return 0;
}
