#include "kernel_generator.h"

using namespace std;

int main(int, char**) {
	cout << kernel_from_string("gauss:12.4")(10, 10).shape()[0] << endl;
	cout << kernel_from_string("box:5")(10, 10).shape()[0] << endl;
	return EXIT_SUCCESS;
}
