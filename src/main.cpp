/*
 * main.cpp
 *
 *  Created on: 07.03.2011
 *      Author: pmarnitz
 */
#include "iostream"
#include "matlab_io.h"
#include "smre_data_double.h"

using namespace std;

int main(int argc, char **argv) {

//	matlab_io io_object;
//	std::vector<double> *in_vec;
//	int no_of_datasets_read, cur_size;
//
//	in_vec = io_object.read_infile("test.mat", no_of_datasets_read);
//
//
//	cout << endl << endl << "*************************" << endl;
//	cout << "\tMAIN FUNCTION" << endl;
//	cout << "*************************" << endl << endl;
//
//	cout << "total number of datasets read: " << no_of_datasets_read << endl;
//
//	for (int i=0; i < no_of_datasets_read; i++) {
//		cur_size = in_vec[i].size();
//		cout << "dataset number " << i << " of size " << cur_size << ":" << endl;
//
//		for (int j = 0; j < cur_size; j++)
//			cout << in_vec[i][j] << " ";
//		cout << endl;
//	}

//	smre_data_double data_object()


//	int myrand;
//
//	for (int asd = 0; asd < 10; asd++) {
//		myrand = (rand() % 50)+1;
//
//		cout << "myrand = " << myrand << endl;
//	}


/*	test_smre_data_double mytest;
	int mytest_res;

	mytest_res = mytest.run(100);

	if (mytest_res == 0)
		cout << "GREAT SUCCESS!!" << endl;
	else
		cout << "I DON'T LIKE!!" << endl;

*/

	matlab_io io_object;

	// c++ style
	{
	vector<smre_data_double> data;
	io_object.read("../matfiles/test.mat", back_inserter(data));

	for(int i = 0 ; i < data.size() ; i++ )
		cout << data[i] << endl;

	io_object.write("test-out1.mat", data.begin(), data.end());
	}

	// c style
	{
	int no_of_datasets_read;
	int *dim_vec;
	smre_data_double* datasets_read;

	datasets_read = io_object.read_file("../matfiles/test.mat", no_of_datasets_read);

	for(int i = 0 ; i < no_of_datasets_read ; i++)
		cout << datasets_read[i] << endl;

	io_object.write_file("test-out.mat", no_of_datasets_read, datasets_read);
	}

	cout << "BYE BYE!" << endl;

	return 0;
}
