/*
 * matlab_io.cpp
 *
 *  Created on: 07.03.2011
 *      Author: pmarnitz
 */

#include <iostream>
#include <vector>
#include <cstring>
#include "matlab_io.h"
//#include "../src/TV/Double3DStorage.h"
//#include "../src/TV/Complex3DStorage.h"
#include "mat.h"
//#include "matrix.h"
//#include "/opt/matlab.local/extern/include/mat.h"
//#include "/opt/matlab.local/extern/include/mat.h"
//#include "/opt/matlab.local/extern/include/matrix.h"
//#include "/opt/matlab.local/extern/include/tmwtypes.h"


//	int read_file(const char* in_file, int no_of_datasets, smre_dataformat *data_read);
smre_data_double* matlab_io::read_file(const char* in_file, int &no_of_datasets) {
	cout << "read a file..." << endl;
	MATFile *pmat;
	mxArray *pa;
	const mwSize *dim_vec;
	double *pr;
	const char **dir;
	const char *name;
	int ndir, i, no_of_dims;
	size_t total_size;

//	smre_data_double *data_double = dynamic_cast<smre_data_double*> (*data_read);
	smre_data_double* data_double;

	std::vector<double> vec_cur;

	pmat = matOpen(in_file, "r");
	if (pmat == NULL) {
		printf("Error opening file %s\n", in_file);
		return 0;
	}

	dir = (const char **)matGetDir(pmat, &ndir);

	data_double = new smre_data_double[ndir];

	cout << "POINTER = " << data_double << endl;

	no_of_datasets = ndir;

	cout << "Number of variables: " << ndir;

	matClose(pmat);

	pmat = matOpen(in_file, "r");

	/* Read in each array. */
	printf("\nReading in the actual array contents:\n");
	for (i=0; i<ndir; i++) {
		pa = matGetNextVariable(pmat, &name);
		if (pa == NULL) {
			printf("Error reading in file %s\n", in_file);
			return 0;
		}
		/*
		 * Diagnose array pa
		 */
		printf("According to its contents, array %s has %d dimensions\n",
				name, mxGetNumberOfDimensions(pa));
		if (mxIsFromGlobalWS(pa))
			printf("  and was a global variable when saved\n");
		else
			printf("  and was a local variable when saved\n");

		dim_vec = mxGetDimensions(pa);
		no_of_dims = mxGetNumberOfDimensions(pa);
		total_size = mxGetNumberOfElements(pa);

		cout << "no_of_dims = " << no_of_dims << endl;
		cout << "total_size = " << total_size << endl;

		for (uint j = 0; j < no_of_dims; j++) {
			cout << "dim no " << j << " is " << dim_vec[j] << endl;
		}


		pr = mxGetPr(pa);
		vec_cur.clear();
		for (uint j = 0; j < total_size; j++)
			vec_cur.push_back((double) *(pr+j));

		//data_double[i] = smre_data_double(vec_cur,(int*) dim_vec, no_of_dims);
		//data_double[i].set(vec_cur, (int*) dim_vec, no_of_dims);

		cout << "size of vector: " << vec_cur.size() << endl;


		mxDestroyArray(pa);


	}

	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n",in_file);
		return 0;
	}
	printf("Done\n");

	return data_double;



}

int matlab_io::write_file(const char* out_file, int no_of_datasets, smre_dataformat* data_read) {

	cout << "matlab_io::write_file not yet implemented!!!" << endl;
	exit(-1);
}


//int matlab_io::write_outfile(const char* out_file, std::vector<double>* out_vec, int no_of_files) {
//	MATFile *pmat;
//	mxArray *pa;
//
//	pmat = matOpen(out_file, "w");
//
////	for (int i = 0; i < no_of_files; i++) {
////		memcpy((void*) pa, out_vec[i].data(), out_vec[i].size());
////	}
//
//}
