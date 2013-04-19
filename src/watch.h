#ifndef __WATCH_H__
#define __WATCH_H__

#include <chrono>

struct watch {
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	watch() : start(std::chrono::high_resolution_clock::now()) {}
	double operator()() {
		return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
	}
};

#endif
