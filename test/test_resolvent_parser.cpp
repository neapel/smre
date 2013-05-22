#include <iostream>
#include "resolvent.h"

using namespace std;

int main() {
	typedef shared_ptr<resolvent_params<float>> T;
	typedef resolvent_h1_params<float> h1;
	typedef resolvent_l2_params<float> l2;
	bool ok = true;
	ok &= dynamic_pointer_cast<l2>(boost::lexical_cast<T>("l2")) != nullptr;
	ok &= dynamic_pointer_cast<l2>(boost::lexical_cast<T>("L2")) != nullptr;
	ok &= dynamic_pointer_cast<h1>(boost::lexical_cast<T>("h1")) != nullptr;
	ok &= dynamic_pointer_cast<h1>(boost::lexical_cast<T>("H1")) != nullptr;
	ok &= dynamic_pointer_cast<h1>(boost::lexical_cast<T>("h1:0.75"))->delta == 0.75;
	ok &= dynamic_pointer_cast<h1>(boost::lexical_cast<T>("H1 0.000"))->delta == 0;
	return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

