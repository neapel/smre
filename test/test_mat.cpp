#include <iostream>
#include "smre_data_double.h"
#include "matlab_io.h"

using namespace std;


int main(int argc, char **argv) {
	
	if(argc != 2) {
		cerr << "usage: " << argv[0] << " file.mat" << endl;
		return -1;
	}


	// c++ style
	{
		matlab_io io_object;
		vector<smre_data_double> data;
		io_object.read(argv[1], back_inserter(data));

		for(size_t i = 0 ; i < data.size() ; i++ )
			cout << data[i] << endl;

		io_object.write("test-out1.mat", data.begin(), data.end());
	}

	// c style
	{
		matlab_io io_object;

		int no_of_datasets_read;
		smre_data_double* datasets_read;

		datasets_read = io_object.read_file(argv[1], no_of_datasets_read);

		for(int i = 0 ; i < no_of_datasets_read ; i++)
			datasets_read[i].print();

		io_object.write_file("test-out2.mat", no_of_datasets_read, datasets_read);
	}

	return 0;
}
