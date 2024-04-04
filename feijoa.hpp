#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <utility>

#ifdef __x86_64__
#include <immintrin.h>
#define FEIJOA_VECTOR_TYPE __m128i
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#define FEIJOA_VECTOR_TYPE poly64x2_t
#endif

class Feijoa {
#ifdef TRUSTEDHASH_TESTS
  public:
#else
  protected:
#endif
    // Low half stores p(x) + x^64, high half stores x^128 // p(x) + x^64.
    FEIJOA_VECTOR_TYPE low_p_low_x_128_div_p;
    // Low half stores x^128 mod p(x), high half stores x^192 mod p(x).
    FEIJOA_VECTOR_TYPE x_128_x_192;
    // Low half stores x^512 mod p(x), high half stores x^576 mod p(x).
    FEIJOA_VECTOR_TYPE x_512_x_576;

    inline uint64_t init_for_basic_computations(uint64_t coeffs) {
        // Let
        //     a(x) = x^128 + (p(x) << 64),
        // then
        //     x^128 // p(x) + x^64 = a(x) // p(x),
        //     x^128 mod p(x) = a(x) mod p(x),

        // This is unrolled long division of a(x) by p(x). As a(x) has zero low 64 bits, each
        // iteration of long division preserves the invariant that only 64 highest bits are
        // non-zero. This enables us to store these 64 bits in a single 64-bit register acc.
        // Additionally, the quotient also fits in 64 bits and thus can be stored in a 64-bit
        // register.
        uint64_t acc = coeffs;
        uint64_t quotient = 0;

        // At the first iteration, we consume just one bit of a(x). Consuming the bit shifts acc one
        // bit to the left, and whether p(x) is added to the accumulator depends on whether the
        // shifted out bit needs to be zeroed.
        uint64_t to_xor = acc >> 63 ? coeffs : 0;
        acc = (acc << 1) ^ to_xor;
        quotient = (quotient << 1) | (coeffs >> 63);

        // The following 63 iterations are performed in 21 groups of 3. This factor seems to provide
        // the best performance.
        //
        // Here's how the iterations are performed. The goal of long division is to add a factor of
        // p(x) to the accumulator to ensure some of its leading bits are zeroed. In our
        // implementation, the goal is to ensure 3 leading bits are zeroed. For instance,
        // 001[64 bits here] can be zeroed by adding p(x) to the accumulator; in code, this can be
        // expressed as acc ^= coeffs. However, cancelling e.g. 010[64 bits here] by adding x * p(x)
        // might fail if p(x) has a coefficient of x^63; indeed, in this case we're left with
        // 001[64 bits here] instead of 000[64 bits here]. Solving this head-on requires handling
        // the bits in sequence, which is what we're trying to avoid here, because latency is the
        // main reason 64 single-bit iterations are slow.
        //
        // This is why we shall use another algorithm to determine which factor of p(x) to add.
        // Determining the coefficients y_0, y_1, y_2 of the factor y_0 + y_1 x + y_2 x^2 is clearly
        // equivalent to solving the equation
        //     deg (a(x) x^3 + (y_0 + y_1 x + y_2 x^2) p(x)) < 64,
        // which is better expressed as a system:
        //     y_0 + y_1 p[63] + y_2 p[62] = a[61],
        //     y_1 + y_2 p[63]             = a[62],
        //     y_2                         = a[63].
        // The solution is
        //     y_0 = a[61] + a[62] p[63] + a[63] (p[62] + p[63]),
        //     y_1 = a[62] + a[63] p[63],
        //     y_2 = a[63],
        // i.e. the factor to add is
        //     (y_0 + y_1 x + y_2 x^2) p(x) = (
        //         a[61] p(x)
        //         + a[62] (p[63] + x) p(x)
        //         + a[63] (p[62] + p[63] + p[63] x + x^2) p(x)
        //     ).
        // The important part is that this formula is linear in respect to (a[61], a[62], a[63]),
        // and if we precompute the factors
        //     p(x),
        //     (p[63] + x) p(x),
        //     (p[62] + p[63] + p[63] x + x^2) p(x)
        // just once, each iteration can be performed easily. We only care about the factor
        // (mod x^64), so let us define
        //     k2(x) = p(x) mod x^64,
        //     k1(x) = (p[63] + x) p(x) mod x^64,
        //     k0(x) = (p[62] + p[63] + p[63] x + x^2) p(x) mod x^64.
        // These are computed as follows:
        uint64_t k2 = coeffs;
        uint64_t k1 = coeffs << 1;
        k1 ^= coeffs >> 63 ? k2 : 0;
        uint64_t k0 = coeffs << 2;
        k0 ^= coeffs >> 63 ? k1 : 0;
        k0 ^= (coeffs >> 62) & 1 ? k2 : 0;
        for (int i = 0; i < 21; i++) {
            // Each iteration then reduces to simply:
            uint64_t xor1 = acc >> 63 ? k0 : 0;
            uint64_t xor2 = (acc >> 62) & 1 ? k1 : 0;
            uint64_t xor3 = (acc >> 61) & 1 ? k2 : 0;
            // We avoid computing the real quotient for now, opting to just collect
            // a[61], a[62], a[63] for now. We'll compute the quotient later with SWAR.
            quotient = (quotient << 3) | (acc >> 61);
            acc = (acc << 3) ^ (xor1 ^ xor2 ^ xor3);
        }
        // The only difficult part left is to restore the quotient. This amounts to computing the
        // factors (y_2, y_1, y_0) given (a[63], a[62], a[61]). The simplest way is to use the
        // sequence
        //     y_2 = a[63],
        //     y_1 = a[62] + y_2 p[63],
        //     y_0 = a[61] + y_1 p[63] + y_2 p[62]
        // to transform
        //     t(x) = y_2 x^2 + a[62] x + a[61]
        // via
        //     t'(x) = t(x) + t[2] (p[63] x + p[62]) = y_2 x^2 + y_1 x + a[61] + y_2 p[62]
        // to
        //     t''(x) = t'(x) + t'[1] p[63] = y_2 x^2 + y_1 x + y_0.
        // These transformations can be trivially applied to all 21 groups at once via SWAR.
        quotient ^= ((quotient >> 2) & 0x1249249249249249) * (coeffs >> 62);
        quotient ^= ((quotient >> 1) & 0x1249249249249249) * (coeffs >> 63);

        low_p_low_x_128_div_p = polynomial_pair(coeffs, quotient);

        uint64_t x_128 = acc;
        uint64_t x_192 = reduce(polynomial_pair(0, x_128));
        x_128_x_192 = polynomial_pair(x_128, x_192);
        return x_128;
    }

    static inline FEIJOA_VECTOR_TYPE polynomial_pair(uint64_t low, uint64_t high) {
#ifdef __x86_64__
        return _mm_set_epi64x(high, low);
#else
        return vcombine_p64(vcreate_p64(low), vcreate_p64(high));
#endif
    }

    static inline FEIJOA_VECTOR_TYPE polynomial_load_unaligned(const FEIJOA_VECTOR_TYPE *p) {
#ifdef __x86_64__
        return _mm_loadu_si128(p);
#else
        return vreinterpretq_p64_s8(vld1q_s8(reinterpret_cast<const int8_t *>(p)));
#endif
    }

    static inline FEIJOA_VECTOR_TYPE polynomial_multiply_low(FEIJOA_VECTOR_TYPE a,
                                                             FEIJOA_VECTOR_TYPE b) {
#ifdef __x86_64__
        return _mm_clmulepi64_si128(a, b, 0x00);
#else
        poly64x2_t result;
        asm("pmull %0.1q, %1.1d, %2.1d" : "=w"(result) : "w"(a), "w"(b));
        return result;
#endif
    }

    static inline FEIJOA_VECTOR_TYPE polynomial_multiply_high(FEIJOA_VECTOR_TYPE a,
                                                              FEIJOA_VECTOR_TYPE b) {
#ifdef __x86_64__
        return _mm_clmulepi64_si128(a, b, 0x11);
#else
        poly64x2_t result;
        asm("pmull2 %0.1q, %1.2d, %2.2d" : "=w"(result) : "w"(a), "w"(b));
        return result;
#endif
    }

    static inline FEIJOA_VECTOR_TYPE polynomial_multiply_low_high(FEIJOA_VECTOR_TYPE a,
                                                                  FEIJOA_VECTOR_TYPE b) {
#ifdef __x86_64__
        return _mm_clmulepi64_si128(a, b, 0x10);
#else
        return polynomial_multiply_high(vdupq_laneq_p64(a, 0), b);
#endif
    }

    static inline FEIJOA_VECTOR_TYPE polynomial_add(FEIJOA_VECTOR_TYPE a, FEIJOA_VECTOR_TYPE b) {
#ifdef __x86_64__
        return _mm_xor_si128(a, b);
#else
        return vaddq_p64(a, b);
#endif
    }

    static inline uint64_t polynomial_low(FEIJOA_VECTOR_TYPE a) {
#ifdef __x86_64__
        return _mm_cvtsi128_si64(a);
#else
        return vgetq_lane_u64(vreinterpretq_u64_p64(a), 0);
#endif
    }

    static inline FEIJOA_VECTOR_TYPE polynomial_zero_high(FEIJOA_VECTOR_TYPE a) {
#ifdef __x86_64__
        return _mm_move_epi64(a);
#else
        return vsetq_lane_p64(0, a, 1);
#endif
    }

    static inline bool has_pdep() {
#ifdef __x86_64__
        return __builtin_cpu_supports("bmi2");
#else
        return false;
#endif
    }

    inline void init_for_hashing(uint64_t x_128) {
        auto x_256 = square(x_128);
        uint64_t x_512 = reduce(square(x_256));
        uint64_t x_576 = reduce(polynomial_pair(0, x_512));
        x_512_x_576 = polynomial_pair(x_512, x_576);
    }

    // Generates a random irreducible polynomial of degree 64 using the given random bit generator.
    template <typename Generator, typename UsePdep>
    static Feijoa random(Generator &generator, UsePdep use_pdep) {
        std::uniform_int_distribution<uint64_t> rng;
        while (true) {
            // A random p(x) is irreducible with probability around 1/64. Forcing
            // (p(x), x(x+1)) = 1 decreases the search space four-fold, thus increasing the
            // probability to 1/16. This yields the expected number of iterations of 16. This is
            // still a lot. As the irreducability check is intrinsically sequential and does not
            // utilize 100% of CPU resources at any tick, it is reasonable to run several checks in
            // lockstep to fill the pipeline better. 6/4 seem to be the most optimal factors with
            // the current performance of Feijoa::Feijoa and is_quasi_irreducible_parallel,
            // depending on whether BMI2 is available.

            std::array<Feijoa, use_pdep ? 6 : 4> feijoas;
            std::array<uint64_t, use_pdep ? 6 : 4> x_128;
            for (size_t i = 0; i < feijoas.size(); i++) {
                x_128[i] = feijoas[i].init_for_basic_computations(fixup_coprime(rng(generator)));
            }
            auto results = is_quasi_irreducible_parallel(feijoas, use_pdep);
            for (size_t i = 0; i < feijoas.size(); i++) {
                auto [quasi_irreducible, payload] = results[i];
                if (quasi_irreducible && feijoas[i].is_really_irreducible(payload)) {
                    feijoas[i].init_for_hashing(x_128[i]);
                    return feijoas[i];
                }
            }
        }
    }

    // Given uniformly randomly chosen p(x), emit uniformly randomly chosen p'(x) such that
    // (p'(x), x * (x + 1)) = 1.
    inline static uint64_t fixup_coprime(uint64_t coeffs) {
        coeffs |= 1;
        coeffs ^= __builtin_parityl(coeffs) << 1;
        return coeffs;
    }

    // Given a(x), computes a representative of a(x)^2 (mod p(x)).
    inline FEIJOA_VECTOR_TYPE square(FEIJOA_VECTOR_TYPE a) const {
        // In F_2,
        //     ((high << 64) + low)^2 = (high^2 << 128) + low^2.
        return polynomial_add(shift_128(polynomial_multiply_high(a, a)),
                              polynomial_multiply_low(a, a));
    }

    // Given a(x), computes a representative of a(x)^2 (mod p(x)).
    template <typename UsePdep> FEIJOA_VECTOR_TYPE square(uint64_t a, UsePdep use_pdep) const {
        if constexpr (use_pdep) {
#ifdef __x86_64__
            uint64_t high, low;
            asm("pdep %1, %2, %0" : "=r"(high) : "r"(0x5555555555555555), "r"(a >> 32));
            asm("pdep %1, %2, %0" : "=r"(low) : "r"(0x5555555555555555), "r"(a));
            return polynomial_pair(low, high);
#else
            __builtin_trap();
#endif
        } else {
            auto a_vec = polynomial_pair(a, 0);
            return polynomial_multiply_low(a_vec, a_vec);
        }
    }

    // Given a(x), computes a representative of a(x)^2 (mod p(x)).
    inline FEIJOA_VECTOR_TYPE square(uint64_t a) const { return square(a, std::false_type{}); }

    // Given a(x), computes a representative of a(x) * x^128 (mod p(x)).
    inline FEIJOA_VECTOR_TYPE shift_128(FEIJOA_VECTOR_TYPE a) const {
        // In F_2,
        //     ((high << 64) + low) << 128 = (high << 192) + (low << 128),
        // hence
        //     a << 128 = high * ((1 << 192) mod p(x)) + low * ((1 << 128) mod p(x)) (mod p(x)).
        return polynomial_add(polynomial_multiply_low(a, x_128_x_192),
                              polynomial_multiply_high(a, x_128_x_192));
    }

    // Given a(x), computes a representative of a(x) * x^512 (mod p(x)).
    inline FEIJOA_VECTOR_TYPE shift_512(FEIJOA_VECTOR_TYPE a) const {
        // In F_2,
        //     ((high << 64) + low) << 512 = (high << 576) + (low << 512),
        // hence
        //     a << 512 = high * ((1 << 576) mod p(x)) + low * ((1 << 512) mod p(x)) (mod p(x)).
        return polynomial_add(polynomial_multiply_low(a, x_512_x_576),
                              polynomial_multiply_high(a, x_512_x_576));
    }

    // Given a(x), computes a(x) mod p(x).
    inline uint64_t reduce(FEIJOA_VECTOR_TYPE a) const {
        // a(x) mod p(x) = ((high << 64) + low) mod p(x) = (high << 64) mod p(x) + low.

        // Perform Barrett reduction on (high << 64): we shall compute
        //     q(x) = (high << 64) // p(x)
        // as
        //     q'(x) = high * (x^128 // p(x)) // x^64.
        // Indeed, replacing truncating division with rational division yields
        //     q(x) = ((high << 64) + A(x)) / p(x),
        //     q'(x) = (high * (x^128 + B(x)) / p(x) + C(x)) / x^64,
        // where deg A(x), deg B(x), deg C(x) < 64, and hence
        //     q(x) = (high << 64) / p(x) + A(x) / p(x),
        //     q'(x) = (high << 64) / p(x) + (high * B(x) / p(x) + C(x)) / x^64,
        // which implies
        //     q(x) + q'(x) = A(x) / p(x) + (high * B(x) / p(x) + C(x)) / x^64,
        // which is a ratio of a negative degree but also a polynomial, i.e. 0.
        auto garbage_q = polynomial_add(a, polynomial_multiply_high(a, low_p_low_x_128_div_p));

        // Compute the low 64 bits of
        //     a(x) mod p(x) = a(x) + q(x) * p(x).
        return polynomial_low(
            polynomial_add(a, polynomial_multiply_low_high(low_p_low_x_128_div_p, garbage_q)));
    }

    // p(x) of degree d is irreducible over F_2 iff:
    //     x^(2^d) = x (mod p(x)), and
    //     for all prime m | d: (x^(2^(d/m)) + x, p(x)) = 1.
    // For d = 64, this translates to
    //     x^(2^64) = x (mod p(x)), and
    //     (x^(2^32) + x, p(x)) = 1.
    // The latter check passes with high probability given that the former check passes.

    // Tests if multiple p(x) pass a loose irreducibility check and returns initialization data for
    // a real check.
    template <size_t N, typename UsePdep>
    static std::array<std::pair<bool, uint64_t>, N>
    is_quasi_irreducible_parallel(const std::array<Feijoa, N> &feijoas, UsePdep use_pdep) {
        // Wrap __m128i in a struct so that the template argument of std::conditional_t does not
        // have attributes that would be dropped, e.g. alignment. See:
        // - https://gcc.gnu.org/bugzilla/show_bug.cgi?id=97222
        // - https://bugs.llvm.org/show_bug.cgi?id=47674
        struct wrapped_vector_t {
            FEIJOA_VECTOR_TYPE wrapped;
        };
        using element_type = typename std::conditional<use_pdep, uint64_t, wrapped_vector_t>::type;

        element_type x_2_32[N];

        // Compute x^(2^32) via repeated squaring.

        // Initialize with x^(2^7) mod p(x).
        for (size_t i = 0; i < N; i++) {
            if constexpr (use_pdep) {
                x_2_32[i] = polynomial_low(feijoas[i].x_128_x_192);
            } else {
                x_2_32[i].wrapped = polynomial_zero_high(feijoas[i].x_128_x_192);
            }
        }

        for (int i = 7; i < 32; i++) {
            for (size_t i = 0; i < N; i++) {
                if constexpr (use_pdep) {
                    x_2_32[i] = feijoas[i].reduce(feijoas[i].square(x_2_32[i], use_pdep));
                } else {
                    x_2_32[i].wrapped = feijoas[i].square(x_2_32[i].wrapped);
                }
            }
        }

        // Compute x^(2^64) as (x^(2^32))^(2^32)).
        element_type x_2_64[N];
        for (size_t i = 0; i < N; i++) {
            x_2_64[i] = x_2_32[i];
        }
        for (int i = 32; i < 64; i++) {
            for (size_t i = 0; i < N; i++) {
                if constexpr (use_pdep) {
                    x_2_64[i] = feijoas[i].reduce(feijoas[i].square(x_2_64[i], use_pdep));
                } else {
                    x_2_64[i].wrapped = feijoas[i].square(x_2_64[i].wrapped);
                }
            }
        }

        std::array<std::pair<bool, uint64_t>, N> results;
        for (size_t i = 0; i < N; i++) {
            // x^(2^64) = x?
            if constexpr (use_pdep) {
                results[i] = {x_2_64[i] == 2, x_2_32[i]};
            } else {
                bool success = feijoas[i].reduce(x_2_64[i].wrapped) == 2;
                results[i] = {success, success ? feijoas[i].reduce(x_2_32[i].wrapped) : 0};
            }
        }
        return results;
    }

    // Tests if p(x) is quasi-irreducible.
    template <typename UsePdep>
    std::pair<bool, uint64_t> is_quasi_irreducible(UsePdep use_pdep) const {
        return is_quasi_irreducible_parallel(std::array<Feijoa, 1>{*this}, use_pdep)[0];
    }

    // Tests if p(x) is quasi-irreducible.
    inline std::pair<bool, uint64_t> is_quasi_irreducible() const {
        if (has_pdep()) {
            return is_quasi_irreducible(std::true_type{});
        } else {
            return is_quasi_irreducible(std::false_type{});
        }
    }

    // Tests if quasi-irreducible p(x) is irreducible.
    inline bool is_really_irreducible(uint64_t x_2_32) const {
        // We now wish to compute (a(x), p(x)), where a(x) = x^(2^32) + x.
        uint64_t a = x_2_32 ^ 2;
        uint64_t coeffs = get_seed();
        // Perform one iteration of the binary Euclidian algorithm to ensure that the second
        // argument fits in 64 bits.
        if (a == 0 || !(coeffs & 1)) {
            return false;
        }
        a >>= __builtin_ctzl(a);
        // p(x) and a(x) are now both odd, compute b(x) = (p(x) + a(x)) / x.
        uint64_t b = (uint64_t{1} << 63) | ((coeffs ^ a) >> 1);
        // Perform the rest of the binary Euclidian algorithm. The invariant is that at the start of
        // each iteartion, a is odd and b is non-zero.
        while (a != 1) {
            // Make b odd.
            b >>= __builtin_ctzl(b);
            if (a > b) {
                std::swap(a, b);
            }
            b ^= a;
            if (b == 0) {
                // GCD is a(x) != 1.
                return false;
            }
        }
        // GCD is (a(x), b(x)) = (1, b(x)) = 1.
        return true;
    }

    // Tests if p(x) is irreducible.
    template <typename UsePdep> bool is_irreducible(UsePdep use_pdep) const {
        auto results = is_quasi_irreducible_parallel(std::array<Feijoa, 1>{*this}, use_pdep);
        auto [quasi_irreducible, payload] = results[0];
        return quasi_irreducible && is_really_irreducible(payload);
    }

  public:
    inline Feijoa() = default;

    // coeffs represents p(x) = x^64 + coeffs_63 x^63 + ... + coeffs_0.
    inline explicit Feijoa(uint64_t coeffs) {
        init_for_hashing(init_for_basic_computations(coeffs));
    }

    // Generates a random irreducible polynomial of degree 64 using the given random bit generator.
    template <typename Generator> static Feijoa random(Generator &generator) {
        if (has_pdep()) {
            return random(generator, std::true_type{});
        } else {
            return random(generator, std::false_type{});
        }
    }

    // Generates several distinct random irreducible polynomials of degree 64 using the given random
    // bit generator.
    template <size_t N, typename Generator>
    static std::array<Feijoa, N> random_many(Generator &generator) {
        std::array<Feijoa, N> feijoas;
        for (auto it = feijoas.begin(); it != feijoas.end(); ++it) {
            do {
                *it = random(generator);
                // In the unlikely case that two polynomials are equal, regenerate. This is a
                // security feature: the probability that two given strings are congruent modulo
                // irreducible polynomials p1(x), ..., pk(x) increases significantly if any of them
                // match.
                //
                // For distinct p1(x), ..., pk(x), though, the collision probability is provably
                // exponential in k. Indeed,
                //     a(x) = 0 (mod pi(x)) for all i
                // holds if p1(x), ..., pk(x) are all present in the factorization of a(x). The
                // worst case is when a(x) is a product of (deg a(x) / 64) distinct irreducible
                // polynomials of degree 64 each. In this case, there are "(deg a(x) / 64) choose k"
                // choices of the set {p1(x), ..., pk(x)} that result in a collision. That's among
                // "2^58 choose k" ways to choose such a set in total. This results in a collision
                // probability of:
                //     ((deg a(x) / 64) choose k) / (2^58 choose k)
                //     ~ ((deg a(x) / 64) choose k) / (2^(58k) / k!)
                // For the common case where deg a(x) / 64 is significantly larger than k, this can
                // be approximated as
                //     ((deg a(x) / 64)^k / k!) / (2^(58k) / k!)
                //     = (deg a(x) / 2^64)^k,
                // which proves that the probability is exponential in k.
                //
                // Keep this quadratic: that should be more efficient in the common case where k is
                // low.
            } while (std::find(feijoas.begin(), it, *it) != it);
        }
        return feijoas;
    }

    // Tests if p(x) is irreducible.
    inline bool is_irreducible() const {
        if (has_pdep()) {
            return is_irreducible(std::true_type{});
        } else {
            return is_irreducible(std::false_type{});
        }
    }

    // Returns a seed that can be used to restore the Feijoa instance later.
    inline uint64_t get_seed() const { return polynomial_low(low_p_low_x_128_div_p); }

    // Computes the hash of an array whose length is a product of 16.
    inline uint64_t reduce(const char *data, size_t n) const {
        assert(n % 16 == 0);

        // Save for reordering of some bits, this function effectively computes
        //     (a_0 + a_1 x + ... + a_{8n-1} x^{n-1} + x^{8n+64}) mod p(x).
        // Adding x^{8n+64} ensures that P(hash(a(x)) = b(x)) is low for fixed b(x), which is a
        // requirement typically imposed on ideal hashes. Indeed,
        //     a(x) + x^{8n+64} = b(x)  (mod p(x))
        // is equivalent to
        //     p(x) | a(x) + x^{8n+64} + b(x),
        // where the right-hand side is neceessarily non-zero. Omitting x^{8n+64} would break this
        // guarantee if deg a(x) < 64. Using a power lower than 64 would break it for n = 0, which
        // is perhaps not a bad thing, really, but using 64 is free so why not.

        auto casted = reinterpret_cast<const FEIJOA_VECTOR_TYPE *>(data);

        size_t i = 0;

        auto acc3 = polynomial_pair(0, 0);
        auto acc2 = polynomial_pair(0, 0);
        auto acc1 = polynomial_pair(0, 0);
        auto acc0 = polynomial_pair(0, 1);
        while (i + 64 <= n) {
            acc3 = polynomial_add(shift_512(acc3), polynomial_load_unaligned(casted + i / 16));
            acc2 = polynomial_add(shift_512(acc2), polynomial_load_unaligned(casted + i / 16 + 1));
            acc1 = polynomial_add(shift_512(acc1), polynomial_load_unaligned(casted + i / 16 + 2));
            acc0 = polynomial_add(shift_512(acc0), polynomial_load_unaligned(casted + i / 16 + 3));
            i += 64;
        }
        auto acc = acc3;
        acc = polynomial_add(shift_128(acc), acc2);
        acc = polynomial_add(shift_128(acc), acc1);
        acc = polynomial_add(shift_128(acc), acc0);

        while (i + 16 <= n) {
            acc = polynomial_add(shift_128(acc), polynomial_load_unaligned(casted + i / 16));
            i += 16;
        }

        return reduce(acc);
    }

    inline bool operator==(const Feijoa &rhs) const { return get_seed() == rhs.get_seed(); }
    inline bool operator!=(const Feijoa &rhs) const { return !(*this == rhs); }
};
