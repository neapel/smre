#ifndef __CONSTRAINT_PARSER_H__
#define __CONSTRAINT_PARSER_H__

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


// parse a list expression:
//   <expr> = <pow> | <expr>;<expr>
//   <pow> = <list> | <base>^<list> | <base>**<list>
//   <list> = {<number>,}<number>
//          | <number:start>,[<number:next>,]...,<number:end>
std::vector<size_t> list_expression(std::string expr) {
	using namespace boost;
	using namespace std;
	vector<size_t> expr_l;
	regex r_pow("((?<base>\\d+)(\\^|\\*\\*))?(?<start>\\d+)((?<list>(,(\\d+))+)|((,(?<next>\\d+))?,?\\.{2,},?(?<end>\\d+)))?");
	regex r_expr("[;:+]"), r_list("\\d+");
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
	return out;
}


#ifdef BOOST_PROGRAM_OPTIONS_VERSION
struct sizes_t : std::vector<size_t> {
	sizes_t() : std::vector<size_t>() {}
	sizes_t(std::initializer_list<size_t> l) : std::vector<size_t>(l) {}
	sizes_t(std::vector<size_t> l) : std::vector<size_t>(l) {}
};

void validate(boost::any &v, const std::vector<std::string> &values, sizes_t *, int) {
	using namespace boost::program_options;
	//?? validators::check_first_occurence(v);
	auto str = validators::get_single_string(values);
	try {
		auto lst = list_expression(str);
		v = boost::any(sizes_t(lst));
	} catch(...) {
		throw validation_error(validation_error::invalid_option_value);
	}
}
#endif

#endif
