/*
 * smre_data_double.h
 *
 *  Created on: 09.03.2011
 *      Author: pmarnitz
 */
#ifndef SMRE_DATA_DOUBLE_H_
#define SMRE_DATA_DOUBLE_H_

#include "../abstract/smre_dataformat.h"
#include <vector>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

class smre_data_double : public smre_dataformat {
private:
	std::vector<double> data;
	int* dims;
	int no_of_dims;

	inline void check_ind(int* ind) {
		for (int i=0; i<no_of_dims; i++) {
			if (ind[i] < 0 || ind[i] >= dims[i]) {
				cout << "Bad index for SMRE_data object!" << endl;
				exit(-1);
			}
		}
	};
public:
	inline int array_index(const int* ind) {
		int array_ind = 0, dim_prod = 1;
		for (int i=0; i<no_of_dims; i++) {
			array_ind += ind[i]*dim_prod;
			dim_prod *= dims[i];
		}
		return array_ind;
	};

	smre_data_double(std::vector<double> data_, int* dims_, int no_of_dims_) : data(data_), dims(dims_), no_of_dims(no_of_dims_) {	};

	smre_data_double() {};

	//abstract virtual functions from smre_dataformat
	int* get_dimensions() {return dims;};

	int get_no_of_dimensions() {return no_of_dims;};

	double &operator() (int* ind) {return data[array_index(ind)]; };

	//TODO: Write print function!
	void print() {cout << "print not yet implemented for smre_data_double!" << endl; };




};















#endif /* SMRE_DATA_DOUBLE_H_ */


