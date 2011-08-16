/*
 * matlab_io.h
 *
 *  Created on: 07.03.2011
 *      Author: pmarnitz
 */
#ifndef MATLAB_IO_H_
#define MATLAB_IO_H_


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "../abstract/smre_io.h"
#include "smre_data_double.h"
//#include "../abstract/smre_dataformat.h"

using namespace std;
//#include "/opt/matlab.local/extern/include/mat.h"
//#include "/opt/matlab.local/extern/include/matrix.h"

class matlab_io : public smre_io {

public:

	static char *infile_name;
	static char *outfile_name;


	//virtual functions from abstract class
	smre_data_double* read_file(const char* in_file, int &no_of_datasets);

	int write_file(const char* out_file, int no_of_datasets, smre_dataformat *data_read);

};


#endif /* MATLAB_IO_H_ */
