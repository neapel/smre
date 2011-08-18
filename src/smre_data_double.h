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
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>


class smre_data_double : public smre_dataformat {
private:
	std::vector<double> data;
	int* dims;
	int no_of_dims;
	std::string _name;

	inline void check_ind(int* ind) {
		for (int i=0; i<no_of_dims; i++)
			if (ind[i] < 0 || ind[i] >= dims[i])
				throw std::runtime_error("Bad index for SMRE_data object!");
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

	template<class I_data, class I_dims>
	smre_data_double(
		std::string name,
		I_data data_begin, I_data data_end,
		I_dims dims_begin, I_dims dims_end
	) : data(), dims(NULL), no_of_dims(0), _name(name) {
		std::copy(data_begin, data_end, std::back_inserter(data));
		//for(I_data i = data_begin ; i != data_end ; i++)
		//	data.push_back( static_cast<double>(*i) );

		no_of_dims = dims_end - dims_begin;
		dims = new int[no_of_dims];
		std::copy(dims_begin, dims_end, dims);
	}

	smre_data_double() {};

	const std::string &name() const { return _name; }

	//abstract virtual functions from smre_dataformat
	int* get_dimensions() {return dims;};

	int get_no_of_dimensions() {return no_of_dims;};

	double &operator() (int* ind) {return data[array_index(ind)]; };

	void print();

	friend std::ostream &operator<<(std::ostream &, smre_data_double &);
};


#endif /* SMRE_DATA_DOUBLE_H_ */
