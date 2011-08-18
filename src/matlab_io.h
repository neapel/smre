/*
 * matlab_io.h
 *
 *  Created on: 07.03.2011
 *      Author: pmarnitz
 */
#ifndef MATLAB_IO_H_
#define MATLAB_IO_H_

#include <matio.h>
#include <string>
#include <vector>
#include "config.h"
#include "../abstract/smre_io.h"
#include "smre_data_double.h"


class matlab_io : public smre_io {

public:
	// Writes arrays from the file to the OutputIterator<smre_data_double>.
	template<class It>
	void read(std::string filename, It out) {
		using namespace std;

		mat_t *mat = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
		if(mat == NULL)
			throw runtime_error("Couldn't open MAT file for reading.");

		// Read all variables
		matvar_t *var = NULL;
		while((var = Mat_VarReadNext(mat)) != NULL) {
			string name(var->name);

			vector<int> dims;
			int count = 1;
			for(int i = 0 ; i < var->rank ; i++) {
				const int dim = var->dims[i];
				dims.push_back(dim);
				count *= dim;
			}

			switch(var->class_type) {
				case MAT_C_DOUBLE: { // double array
					double *data = reinterpret_cast<double *>(var->data);
					*out++ = smre_data_double(name, &data[0], &data[count], dims.begin(), dims.end());
				} break;
				case MAT_C_SINGLE: { // float array
					float *data = reinterpret_cast<float *>(var->data);
					*out++ = smre_data_double(name, &data[0], &data[count], dims.begin(), dims.end());
				}
				default:
					// ignore.
					break;
			}
		}

		if(Mat_Close(mat) != 0)
			throw runtime_error("Couldn't close MAT file.");
	}


	// Writes arrays from the Iterator<smre_data_double> to the file.
	template<class It>
	void write(std::string filename, It begin, It end) {
		using namespace std;
		mat_t *mat = Mat_Create(filename.c_str(), "SMRE " VERSION " (libmatio writer)");
		if(mat == NULL)
			throw runtime_error("Couldn't create new MAT file.");

		// Write all variables
		for(It arr = begin ; arr != end ; arr++) {
			// total number of cells
			int count = 1;
			const int rank = arr->get_no_of_dimensions();
			int *dims = arr->get_dimensions();
			for(int i = 0 ; i < rank ; i++)
				count *= dims[i];

			// copy data
			double *data = new double[count];
			int *index = new int[rank];
			for(int i = 0 ; i < rank ; i++)
				index[i] = 0;
			for(int i = 0 ; i < count ; i++) {
				index[0] = i;
				data[i] = (*arr)(index);
			}
			delete [] index;

			// create variable header
			const char *name = arr->name().c_str();
			matvar_t *var = Mat_VarCreate(name, MAT_C_DOUBLE, MAT_T_DOUBLE, rank, dims, data, MEM_CONSERVE);
			if(var == NULL)
				throw runtime_error("Couldn't create variable.");

			if(Mat_VarWrite(mat, var, /*compress*/false) != 0)
				throw runtime_error("Couldn't write variable.");

			Mat_VarFree(var);
			delete [] data;
		}

		if(Mat_Close(mat) != 0)
			throw runtime_error("Couldn't close MAT file after writing.");
	}


	// old interface:
	smre_data_double* read_file(const char* in_file, int &no_of_datasets);
	int write_file(const char* out_file, int no_of_datasets, smre_dataformat *data_read);
};


#endif /* MATLAB_IO_H_ */
