.PHONY: benchmark test format

ARCH := $(shell uname -m)

CXXOPTS := -O2 -std=c++17 -g -Wall
ifeq ($(ARCH),x86_64)
CXXOPTS += -mpclmul
endif
ifeq ($(ARCH),aarch64)
CXXOPTS += -mcpu=generic+crypto
endif

all:
	echo "This Makefile is not supposed to be used for builds."

benchmark: target/benchmark
	target/benchmark
target/benchmark: benchmark.cpp feijoa.hpp
	mkdir -p target
	$(CXX) benchmark.cpp -o target/benchmark $(CXXOPTS)

test: target/test
	target/test
target/test: test.cpp feijoa.hpp
	mkdir -p target
	$(CXX) test.cpp -o target/test $(CXXOPTS)

format:
	clang-format-17 -i benchmark.cpp test.cpp feijoa.hpp
