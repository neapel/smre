#include "operator_split.h"
#include <string>
#include <tuple>
#include <iostream>
#include <cassert>
#include "debug_tools.h"
using namespace std;


double f2(float a_new, float a_old, double b_old) {
	cout << "f2(new: " <<  a_new << ", old: " << a_old << ", " << b_old << ")";
	auto x = a_new + a_old + b_old + 1;
	cout << " = " << x << endl;
	return x; // b_new
}


int main(int, char**) {
	
	int count = 0;

	auto v = operator_split(
		// initial values:
		tuple<float, double>(0.0f, 0.0),
		// functions
		make_tuple(
			// using an inline function:
			[](float a_old, double b_old) { // ne == old
				cout << "f1(old: " << a_old << ", " << b_old << ")";
				auto x = a_old + b_old + 1;
				cout << " = " << x << endl;
				return x; // a_new
			},
			// using a function pointer:
			f2
		),
		// return true to continue
		function<bool(float,double)>( [&count](float a, double b) { // use `count` from context
			cout << "step " << count << ": ";
			count++;
			if(count > 1000) return false;
			cout << "stop?(" << a << ", " << b << ")" << endl;
			return a + b < 500;
		} )
	);
	cout << "out = " << v << endl;

	assert(v == make_tuple(364.0f, 728.0));

	return 0;
}
