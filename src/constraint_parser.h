#ifndef __CONSTRAINT_PARSER_H__
#define __CONSTRAINT_PARSER_H__

#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>

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


// parse a list expression:
//   <expr> = <pow> | <expr>;<expr>
//   <pow> = <list> | <base>^<list> | <base>**<list>
//   <list> = {<number>,}<number>
//          | <number:start>,[<number:next>,]...,<number:end>
std::vector<size_t> list_expression(std::string expr) {
	using namespace boost;
	using namespace std;
	vector<size_t> expr_l;
	regex r_pow(R"(\s*((?<base>\d+)\s*(\^|\*\*)\s*)?(?<start>\d+)((?<list>(\s*,\s*(\d+))+)|((\s*,\s*(?<next>\d+))?\s*,?\s*\.{2,}\s*,?\s*(?<end>\d+)))?\s*)");
	regex r_expr(R"([;:+])"), r_list(R"(\d+)");
	for(auto s_pow : make_regex_token_iterator(expr, r_expr, -1)) {
		auto s = s_pow.str(); // mandatory??? regex_match(s_pow.str()...) breaks.
		smatch m;
		if(!regex_match(s, m, r_pow))
			throw invalid_argument("invalid list expression.");
		auto start = lexical_cast<size_t>(m["start"]);
		vector<size_t> pow_l;
		pow_l.push_back(start);
		auto base_m = m["base"], list_m = m["list"], next_m = m["next"], end_m = m["end"];
		if(list_m.matched) {
			for(auto i : make_regex_token_iterator(list_m.str(), r_list)) {
				pow_l.push_back(lexical_cast<size_t>(i.str()));
			}
		} else if(end_m.matched) {
			auto next = next_m.matched ? lexical_cast<size_t>(next_m) : start + 1;
			auto end = lexical_cast<size_t>(end_m);
			if(end <= start || next <= start || end <= next)
				throw invalid_argument("invalid list range.");
			auto delta = next - start;
			for(size_t i = next ; i <= end ; i += delta)
				pow_l.push_back(i);
		}
		if(base_m.matched) {
			auto base = lexical_cast<size_t>(base_m);
			for(auto &v : pow_l)
				v = pow(base, v);
		}
		copy(pow_l.begin(), pow_l.end(), back_inserter(expr_l));
	}
	sort(expr_l.begin(), expr_l.end());
	vector<size_t> out;
	unique_copy(expr_l.begin(), expr_l.end(), back_inserter(out));
	if(out.empty()) throw invalid_argument("empty lsit expression.");
	return out;
}

namespace std {

std::istream &operator>>(std::istream &i, std::vector<size_t> &s) {
	std::istream_iterator<char> begin{i}, end;
	const std::string b(begin, end);
	s = list_expression(b);
	return i;
}

std::ostream &operator<<(std::ostream &o, const std::vector<size_t> &s) {
	for(size_t i = 0 ; i < s.size() ; i++) {
		if(i != 0) o << ',';
		o << s[i];
	}
	return o;
}

}

#endif
