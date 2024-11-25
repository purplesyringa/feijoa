# Feijoa

> **Disclaimer:** [CLHash](https://github.com/lemire/clhash) provides better guarantees while performing faster. However, it currently doesn't support ARM and requires SSE4.1, while Feijoa supports ARM NEON and only requires [CLMUL](https://en.wikipedia.org/wiki/CLMUL_instruction_set) on x86. Do with this what you will.

**Feijoa** is a highly efficient hash-like cryptographic primitive.

It provides the same interface that typical salted hash functions provide: you initialize a hasher object with a salt, which then allows you to compute 64-bit hashes of byte objects. However, the security guarantees differ from those of typical hash functions like xxhash or SHA:

- On a positive note, a Feijoa *with an unknown salt* provides *mathematical* guarantees about collisions. This is a good thing, because you don't have to trust a hash function just because it passed SMHasher, and it guards against accidental and intentional backdoors. If the probabilities are too high for your use case, you can create several Feijoa objects with different salts, and the collision rate can be *proved* to decrease exponentially. The guarantees are:
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

If the malicious actor has no direct access to the runtime that uses Feijoa and has no way to learn the salt or the hash values, it can be *proven* that you are safe.


## Usage

Feijoa is a single-header C++17 library (although GCC and Clang compile it in C++11 mode with warnings). It needs to be compiled with `-mpclmul` (on x86_64) or `-mcpu=generic+crypto` (on aarch64). Other architectures are unsupported at the moment.

The library provides a single class `Feijoa` with a static method to construct a new instance of Feijoa with a random seed:

```cpp
std::random_device rd;
std::mt19937_64 generator{rd()};
Feijoa feijoa = Feijoa::random(generator);
```

It is insecure to initialize a `Feijoa` object directly from a randomly generated 64-bit integer via `Feijoa feijoa(generator());`. `Feijoa::random` uses a smarter seed generation method to ensure the security guarantees hold.

The instance can then be used repeatedly to hash arrays:

```cpp
std::array<char, 4096> array{};
uint64_t hash = feijoa.reduce(array.data(), array.size());
```

The length of the array must be divisible by 16.

Hashes provided by different instances of Feijoa must not be compared: no cryptographic guarantees are made about them. However, it *is* a good idea to create a new Feijoa for each separate use, to make sure accidentally leaking a salt or a hash cannot lead to an attack on another part of the program.

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


## How it works

Implementation-wise, Feijoa is just CRC64 with a random irreducible polynomial used as the modulo. Doesn't sound so ingenious anymore, does it? :-)

While using CRC for cryptographic purposes immediately raises alarms, CRC's bad publicity only covers the use cases that Feijoa explicitly avoids. The critical differences are that Feijoa uses a dynamic polynomial and both the hashes and the modulo are kept secret. These conditions are reasonable in many cases, which is why Feijoa thrives where CRC fails.

Basing on CRC also lets Feijoa reuse many optimizations Intel has introduced into the ISA for CRC and other cryptographic operations. For instance, Feijoa uses the `pclmul` instruction to consume 8 bytes from the input stream at once with the highest possible degree of parallelism.

An important thing to notice is that Feijoa puts quite a bit of thought into optimizating the generation of random irreducible polynomials. On my Haswell, Feijoa manages to generate 338k random irreducible polynomials per second, which amounts to 338k instantiations of e.g. a hashmap. This investment into performance is in hope that developers avoid sharing a Feijoa instance between multiple hashtables. Indeed, isolating Feijoas by using exactly one Feijoa per hashtable reduces the scope of attacks if the salt or the hash get leaked.
