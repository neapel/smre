#include "gil_io.h"
#include "multi_array_print.h"
using namespace std;

int main(int argc, char **argv) {
	if(argc != 3) return EXIT_FAILURE;
	
	auto x = read_image(argv[1]);
	cout << x << endl;
	write_image(argv[2], x);
}
