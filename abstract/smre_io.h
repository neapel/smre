/*
 * smre_io.h
 *
 *  Created on: 09.03.2011
 *      Author: pmarnitz
 */

#ifndef SMRE_IO_H_
#define SMRE_IO_H_

class smre_dataformat;

class smre_io{
public:

	virtual smre_dataformat* read_file(const char* in_file, int &no_of_datasets) = 0;

	virtual int write_file(const char* out_file, int no_of_datasets, smre_dataformat *data_read) = 0;

};




#endif /* SMRE_IO_H_ */
