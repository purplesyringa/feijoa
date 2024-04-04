#define TRUSTEDHASH_TESTS
#include "feijoa.hpp"
#include <cassert>

bool eq(__m128i a, __m128i b) { return a[0] == b[0] && a[1] == b[1]; }

int main() {
    for (uint64_t a : std::array<uint64_t, 4>{0xfcf980c83018a6d2, 0x3856dbeaf383ce21,
                                              0xf4d0134ced3abf0e, 0x471608aa92303b90}) {
        uint64_t fixup = Feijoa::fixup_coprime(a);
        assert((fixup >> 2) == (a >> 2));
        assert(fixup & 1);
        assert(!__builtin_parityl(fixup));
    }

    char buffer[] = {102,  -53, -28, 108,  28, 23,  -10,  -77,  117, 32,   121, 9,  12,
                     -87,  15,  -19, -51,  24, -60, -63,  -126, -61, 30,   -72, 14, -10,
                     -108, 12,  105, -102, -6, 8,   -102, 28,   48,  -106, 65,  17, -113,
                     -79,  46,  -61, 102,  68, 67,  -38,  -64,  -75, -115, -31, 41, -63,
                     -11,  44,  -6,  -114, 64, -6,  5,    56,   42,  89,   37,  26};

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
        assert(eq(feijoa.square(_mm_set_epi64x(0, 0b101)), _mm_set_epi64x(0, 0b10001)));
        // (x^32)^2 = x^64
        assert(eq(feijoa.square(_mm_set_epi64x(0, uint64_t{1} << 32)), _mm_set_epi64x(1, 0)));
        // (x^64)^2 = 0
        assert(eq(feijoa.square(_mm_set_epi64x(1, 0)), _mm_setzero_si128()));
        if (__builtin_cpu_supports("bmi2")) {
            // (x^33 + x^32 + x^2 + 1)^2 = x^66 + x^64 + x^4 + 1
            assert(
                eq(feijoa.square(0x300000005, std::true_type{}), _mm_set_epi64x(0b101, 0b10001)));
        }
        assert(eq(feijoa.shift_128(_mm_set_epi64x(123, 456)), _mm_setzero_si128()));
        assert(eq(feijoa.shift_512(_mm_set_epi64x(123, 456)), _mm_setzero_si128()));
        assert(feijoa.reduce(_mm_set_epi64x(123, 456)) == 456);
        assert(feijoa.reduce(buffer, sizeof(buffer)) == 0x8efa2cf5c129e18d);
        assert(feijoa.reduce(buffer + 16, sizeof(buffer) - 16) == 0x8efa2cf5c129e18d);
        assert(feijoa.reduce(buffer, 0) == 0);
        assert(!feijoa.is_irreducible(std::false_type{}));
        assert(!feijoa.is_quasi_irreducible(std::false_type{}).first);
        if (__builtin_cpu_supports("bmi2")) {
            assert(!feijoa.is_irreducible(std::true_type{}));
            assert(!feijoa.is_quasi_irreducible(std::true_type{}).first);
        }
    }

    {
        // x^64 + x^63 + x^62 + x^61 + x^58 + x^54 + x^48 + x^46 + x^43 + x^41 + x^40 + x^39 + x^37
        //     + x^36 + x^35 + x^33 + x^30 + x^28 + x^27 + x^26 + x^25 + x^23 + x^22 + x^21 + x^20
        //     + x^19 + x^8 + x^7 + x^6 + x^2 + 1
        Feijoa feijoa(0xe4414bba5ef801c5);
        assert(feijoa.low_p_low_x_128_div_p[0] == (long long)0xe4414bba5ef801c5);
        assert(feijoa.low_p_low_x_128_div_p[1] == (long long)0x9cd26aeea99afeb4);
        assert(feijoa.x_128_x_192[0] == (long long)0xe62e245859af4764);
        assert(feijoa.x_128_x_192[1] == (long long)0x329ed7d43d59826c);
        assert(feijoa.x_512_x_576[0] == (long long)0xd0b65ca1f87a3466);
        assert(feijoa.x_512_x_576[1] == (long long)0x03cbfc7a304ea1dc);
        assert(feijoa.get_seed() == 0xe4414bba5ef801c5);
        assert(feijoa == feijoa);
        assert(feijoa != Feijoa{1});
        __m128i a = _mm_set_epi64x(0xce061518d88c8a77, 0x7806f7d4cc65b145);
        assert(feijoa.reduce(a) == 0xb2a3023836224dae);
        assert(feijoa.reduce(feijoa.square(a)) == 0x47ff962ca9e606df);
        assert(feijoa.reduce(feijoa.shift_128(a)) == 0x6f010a6522f6d04a);
        assert(feijoa.reduce(feijoa.shift_512(a)) == 0x3c04fde0b1149209);
        if (__builtin_cpu_supports("bmi2")) {
            assert(eq(feijoa.square(0xe30d03531263e5f5, std::true_type{}),
                      _mm_set_epi64x(0x5405005100051105, 0x0104140554115511)));
        }
        assert(!feijoa.is_irreducible(std::false_type{}));
        assert(!feijoa.is_quasi_irreducible(std::false_type{}).first);
        if (__builtin_cpu_supports("bmi2")) {
            assert(!feijoa.is_irreducible(std::true_type{}));
            assert(!feijoa.is_quasi_irreducible(std::true_type{}).first);
        }
        assert(feijoa.reduce(buffer, sizeof(buffer)) == 0x45d383006154c9b7);
        assert(feijoa.reduce(buffer + 16, sizeof(buffer) - 16) == 0xfe98ad376a5e9a75);
        assert(feijoa.reduce(buffer, 0) == 0xe4414bba5ef801c5);
    }

    {
        // x^64 + x^62 + x^61 + x^60 + x^57 + x^56 + x^54 + x^53 + x^48 + x^47 + x^46 + x^45 + x^41
        //     + x^37 + x^35 + x^34 + x^32 + x^31 + x^30 + x^29 + x^28 + x^27 + x^25 + x^19 + x^18
        //     + x^17 + x^8 + x^4 + x^3 + x^2 + 1
        Feijoa feijoa{0x7361e22dfa0e011d};
        assert(feijoa.low_p_low_x_128_div_p[0] == (long long)0x7361e22dfa0e011d);
        assert(feijoa.low_p_low_x_128_div_p[1] == (long long)0x619b551519cfaa41);
        assert(feijoa.x_128_x_192[0] == (long long)0x9822de4a3652b45d);
        assert(feijoa.x_128_x_192[1] == (long long)0xd937bd8ad5c7974d);
        assert(feijoa.x_512_x_576[0] == (long long)0x7585192ed81b3087);
        assert(feijoa.x_512_x_576[1] == (long long)0xee9e707757f4e581);
        assert(feijoa.get_seed() == 0x7361e22dfa0e011d);
        assert(feijoa == feijoa);
        assert(feijoa != Feijoa{1});
        __m128i a = _mm_set_epi64x(0xc17b85927574fab9, 0xeebe4ebfdbf8869d);
        assert(feijoa.reduce(a) == 0x412f5c88a5785e87);
        assert(feijoa.reduce(feijoa.square(a)) == 0x9682819ce543af15);
        assert(feijoa.reduce(feijoa.shift_128(a)) == 0x61017f24d3cfdfec);
        assert(feijoa.reduce(feijoa.shift_512(a)) == 0xd4d2d6bc13a5c724);
        if (__builtin_cpu_supports("bmi2")) {
            assert(eq(feijoa.square(0xc78c0896c394c3cf, std::true_type{}),
                      _mm_set_epi64x(0x5015405000404114, 0x5005411050055055)));
        }
        assert(feijoa.is_irreducible(std::false_type{}));
        assert(feijoa.is_quasi_irreducible(std::false_type{}).first);
        if (__builtin_cpu_supports("bmi2")) {
            assert(feijoa.is_irreducible(std::true_type{}));
            assert(feijoa.is_quasi_irreducible(std::true_type{}).first);
        }
        assert(feijoa.reduce(buffer, sizeof(buffer)) == 0xb2d8e830efef6b49);
        assert(feijoa.reduce(buffer + 16, sizeof(buffer) - 16) == 0x6cc9738b07e91f18);
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
        if (__builtin_cpu_supports("bmi2")) {
            assert(!feijoa.is_irreducible(std::true_type{}));
            assert(feijoa.is_quasi_irreducible(std::true_type{}).first);
        }
    }

    std::mt19937_64 generator{0};
    assert(Feijoa::random(generator, std::false_type{}).is_irreducible());
    if (__builtin_cpu_supports("bmi2")) {
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
