#ifndef __CONSTRAINT_PARSER_H__
#define __CONSTRAINT_PARSER_H__

#include "chambolle_pock.h"
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

namespace std {
	template<class I>
	boost::regex_token_iterator<I> begin(boost::regex_token_iterator<I> &i) {
		return i;
	}

	template<class I>
	boost::regex_token_iterator<I> end(boost::regex_token_iterator<I> &) {
		return {};
	}
}


// parse a kernel list expression: "<kind>:<size0>[,<size1>,...,<sizeN>]"
std::vector<constraint> constraints_from_string(std::string expr) {
	using namespace boost;
	std::vector<constraint> kernels;
	regex r("(?<kind>\\w+):(?<start>\\d+)((?<list>(,(\\d+))+)|((,(?<next>\\d+))?,\\.{2,},(?<end>\\d+)))?");
	smatch m;
	if(regex_match(expr, m, r)) {
		std::vector<size_t> sizes;
		auto start = lexical_cast<size_t>(m["start"]);
		sizes.push_back(start);
		auto list_m = m["list"], next_m = m["next"], end_m = m["end"];
		if(list_m.matched) {
			regex split("\\d+");
			for(auto i : make_regex_token_iterator(list_m.str(), split))
				sizes.push_back(lexical_cast<size_t>(i));
		} else if(end_m.matched) {
			auto next = next_m.matched ? lexical_cast<size_t>(next_m) : start + 1;
			auto end = lexical_cast<size_t>(end_m);
			if(end <= start || next <= start || end <= next)
				throw std::invalid_argument("invalid kernels range.");
			auto delta = next - start;
			for(size_t i = next ; i <= end ; i += delta)
				sizes.push_back(i);
		}

		auto ks = format("%s:%d");
		auto kind = m["kind"].str();
		for(size_t s : sizes)
			kernels.push_back(constraint{str(ks % kind % s)});
		return kernels;
	}
	throw std::invalid_argument("invalid kernels expression.");
}

#endif
