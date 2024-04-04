.PHONY: benchmark test format

CXXOPTS := -O2 -std=c++17 -Wall -mpclmul

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
