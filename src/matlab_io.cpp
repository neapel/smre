#include "matlab_io.h"
#if HAVE_MATIO

using namespace std;


smre_data_double* matlab_io::read_file(const char* in_file, int &no_of_datasets) {
	vector<smre_data_double> n;
	read(string(in_file), back_inserter(n));

	no_of_datasets = n.size();
	smre_data_double *out = new smre_data_double[n.size()];
	copy(n.begin(), n.end(), out);
	return out;
}


int matlab_io::write_file(const char* out_file, int no_of_datasets, smre_dataformat* data_read) {
	smre_data_double *data = dynamic_cast<smre_data_double *>(data_read);
	write(string(out_file), &data[0], &data[no_of_datasets]);
	return 0;
}

#endif
