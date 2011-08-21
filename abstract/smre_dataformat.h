#ifndef SMRE_DATAFORMAT_H_
#define SMRE_DATAFORMAT_H_

#include <iostream>


/*! Multi-dimensional array class.
 * @todo make generic.
 * @author pmarnitz
 */
class smre_dataformat {
public:

	/*! Get the number of elements in each dimension.
	 * @returns Array of rank-length with element counts. Length is get_no_of_dimensions().
	 */
	virtual int* get_dimensions() = 0;

	/*! Get the number of dimensions (rank) of this array.
	 * @returns Number greater 0.
	 */
	virtual int get_no_of_dimensions() = 0;

	/*! Access an element of this array.
	 * @param[in] ind  Array of rank-length with indices.
	 * @returns The element (writable).
	 */
	virtual double &operator() (int* ind) = 0;

	/*! Prints this array's content.
	 * @deprecated use std::ostream-operators.
	 */
	virtual void print() = 0;
};

#endif /* SMRE_DATAFORMAT_H_ */
