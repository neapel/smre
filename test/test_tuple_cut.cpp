#include "tuple_cut.h"
#include <cassert>

using namespace std;

int main(int, char**) {
	tuple<int, float, double, char> t {1,2,3,'x'};
	
	assert((tuple_cut<0>(t) == tuple<>()));
	assert((tuple_cut<1>(t) == tuple<int>(1)));
	assert((tuple_cut<2>(t) == tuple<int, float>(1,2)));
	assert((tuple_cut<3>(t) == tuple<int, float, double>(1,2,3)));
	assert((tuple_cut<4>(t) == tuple<int, float, double, char>(1,2,3,'x')));
}
