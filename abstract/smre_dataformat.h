/*
 * smre_dataformat.h
 *
 *  Created on: 09.03.2011
 *      Author: pmarnitz
 */

#ifndef SMRE_DATAFORMAT_H_
#define SMRE_DATAFORMAT_H_

#include <iostream>


class smre_dataformat {
public:

	virtual int* get_dimensions() = 0;

	virtual int get_no_of_dimensions() = 0;

	//TODO: Rückgabetyp vorerst auf double gesetzt; sollte allgemeinen Typen zulassen
	virtual double &operator() (int* ind) = 0;

	virtual void print() = 0;

};

#endif /* SMRE_DATAFORMAT_H_ */
