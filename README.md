# Feijoa

**Feijoa** is a highly efficient hash-like cryptographic primitive.

It provides the same interface that typical salted hash functions provide: you initialize a hasher object with a salt, which then allows you to compute 64-bit hashes of byte objects. However, the security guarantees differ from those of typical hash functions like xxhash or SHA:

- On a positive note, a Feijoa *with an unknown salt* provides *provable* guarantees about collisions. This is a good thing, because you don't have to trust a hash function just because it passed SMHasher, and it guard against accidental and intentional backdoors. If the probabilities are too high for your use case, you can create several Feijoa objects with different salts, and the collision rate will *provably* decrease exponentially. The guarantees are:
	- Hashes of two maliciously generated N-byte strings collide with probability less than `N * 2^-61`.
	- Hashes of two randomly generated strings collide with probability `2^-64`.
- On a negative note, if the salt or the hash values are ever leaked, generating collisions is trivial.

This means that under certain circumstances Feijoa can be used in place of a *cryptographic* hash. For example, if you wish to hash 4096-byte strings and a collision tolerance of `2^-98` suffices, producing a 128-bit hash via two distinct instances of Feijoa might be used instead of e.g. BLAKE3, while providing better performance.


## Benchmarks

On my Intel Haswell, the hashing performance is:

- Feijoa: 14.3 GiB/s
- wyhash: 24.2 GiB/s
- wyhash+condom: 18.3 GiB/s
- BLAKE3: 2.3 GiB/s


## Checklist

Here's a checklist to see if Feijoa is safe for your use case:

- Do you send hash values to an untrusted party? If yes, Feijoa is not for you.
- Do side-channel attacks apply to you? If yes, Feijoa is *likely* not for you.
- Can timing attacks be performed to determine information about a hash? For instance, can a malicious actor measure the time taken to access an element of a hash table that doesn't fit in cache? If yes, Feijoa is *likely* not for you.

If the malicious actor has no direct access to the runtime that uses Feijoa and has no way to learn the salt or the hash values, you are *provably* safe.


## Usage

Feijoa is a single-header C++17 library (although GCC and Clang compile it in C++11 mode with warnings). It needs to be compiled with `-mpclmul` (on x86_64) or `-mcpu=generic+crypto` (on aarch64). Other architectures are unsupported at the moment.

The library provides a single class `Feijoa` with a static method to construct a new instance of Feijoa with a random seed:

```cpp
std::random_device rd;
std::mt19937_64 generator{rd()};
Feijoa feijoa = Feijoa::random(generator);
```

The instance can then be used repeatedly to hash arrays:

```cpp
std::array<char, 4096> array{};
uint64_t hash = feijoa.reduce(array.data(), array.size());
```

The length of the array must be divisible by 16.

Hashes provided by different instances of Feijoa must not be compared: no cryptographic guarantees are made about them.

A seed can be extracted from a Feijoa instance to re-create it later:

```cpp
Feijoa feijoa1 = Feijoa::random(generator);
uint64_t seed = feijoa1.get_seed();
Feijoa feijoa2{seed};
assert(feijoa1 == feijoa2);
```

If you wish to increase security by increasing hash length, do NOT create `Feijoa` instances separately if they are going to be used together -- this decreases security guarantees! Instead, create all the instances at once:

```cpp
std::array<Feijoa, 2> feijoas = Feijoa::random_many<2>(generator);
```
