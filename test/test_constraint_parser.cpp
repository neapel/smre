#include <iostream>
#include "constraint_parser.h"

using namespace std;

bool range_equal(vector<size_t> v, initializer_list<size_t> e) {
	if(!std::equal(v.begin(), v.end(), e.begin())) {
		for(auto x : v) cerr << x << ",";
		cerr << "  ==  ";
		for(auto x : e) cerr << x << ",";
		cerr << endl;
		return false;
	} else return true;
}

int main() {
	bool r = true;
	r &= range_equal(list_expression("10"), {10});
	r &= range_equal(list_expression("1..5"), {1,2,3,4,5});
	r &= range_equal(list_expression("1,..,5"), {1,2,3,4,5});
	r &= range_equal(list_expression("1,....,5"), {1,2,3,4,5});
	r &= range_equal(list_expression("1,2,7"), {1,2,7});
	r &= range_equal(list_expression("1,2"), {1,2,7});
	r &= range_equal(list_expression("1,3..8"), {1,3,5,7});
	r &= range_equal(list_expression("5,10,...,20"), {5,10,15,20});
	r &= range_equal(list_expression("2**1..4"), {2,4,8,16});
	r &= range_equal(list_expression("1..3;10;2**1,2"), {1,2,3,4,10});
	return r ? EXIT_SUCCESS : EXIT_FAILURE;
}
