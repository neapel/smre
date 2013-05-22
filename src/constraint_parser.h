#ifndef __CONSTRAINT_PARSER_H__
#define __CONSTRAINT_PARSER_H__

#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <boost/any.hpp>

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

struct sizes_t : std::vector<size_t> {
	std::string expr;

	// parse a list expression:
	//   <expr> = <pow> | <expr>;<expr>
	//   <pow> = <list> | <base>^<list> | <base>**<list>
	//   <list> = {<number>,}<number>
	//          | <number:start>,[<number:next>,]...,<number:end>
	sizes_t(std::string expr) : expr(expr) {
		using namespace boost;
		using namespace std;
		vector<size_t> expr_l;
		static regex r_pow(R"(\s*((?<base>\d+)\s*(\^|\*\*)\s*)?(?<start>\d+)((?<list>(\s*,\s*(\d+))+)|((\s*,\s*(?<next>\d+))?\s*,?\s*\.{2,}\s*,?\s*(?<end>\d+)))?\s*)");
		static regex r_expr(R"([;:+])"), r_list(R"(\d+)");
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
		unique_copy(expr_l.begin(), expr_l.end(), back_inserter(*this));
		if(empty()) throw invalid_argument("empty list expression.");
	}

	sizes_t(const std::vector<size_t> &orig) : std::vector<size_t>(orig) {
		make_expr();
	}

	sizes_t(const std::initializer_list<size_t> &orig) : std::vector<size_t>(orig) {
		make_expr();
	}

	sizes_t() {}

	void make_expr() {
		std::ostringstream s;
		for(size_t i = 0 ; i < size() ; i++) {
			if(i != 0) s << ',';
			s << operator[](i);
		}
		expr = s.str();
	}
};

std::istream &operator>>(std::istream &i, sizes_t &s) {
	std::istream_iterator<char> begin{i}, end;
	s = sizes_t(std::string(begin, end));
	i.unget();
	i.clear(); // otherwise lexical_cast fails.
	return i;
}

std::ostream &operator<<(std::ostream &o, const sizes_t &s) {
	return o << s.expr;
}


void validate(boost::any& v, const std::vector<std::string>& s, sizes_t*, int) {
	if(!v.empty() || s.size() != 1) throw std::invalid_argument("value specified twice");
	v = boost::any(sizes_t(s[0]));
}



#endif
