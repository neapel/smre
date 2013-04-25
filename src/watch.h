#ifndef __WATCH_H__
#define __WATCH_H__

#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

template<class It>
typename It::value_type median(It begin, It end) {
	std::sort(begin, end);
	size_t n = end - begin;
	if(n % 2 == 0) return (*(begin + n/2) + *(begin + (n/2 + 1))) / 2;
	else return *(begin + n/2);
}


struct watch {
	std::vector<double> times;
	std::chrono::time_point<std::chrono::high_resolution_clock> begin;
	watch() : times(), begin(std::chrono::high_resolution_clock::now()) {}

	void start() {
		begin = std::chrono::high_resolution_clock::now();
	}
	void stop() {
		auto end = std::chrono::high_resolution_clock::now();
		times.push_back(std::chrono::duration<double>(end - begin).count());
	}

	double median() {
		return ::median(times.begin(), times.end());
	}

	double rel_error() {
		double m = median();
		std::vector<double> diff;
		for(auto t : times)
			diff.push_back(std::abs(t - m));
		double var = 1.4826 * ::median(diff.begin(), diff.end());
		return var / m;
	}

};

#endif
