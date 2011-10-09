#include "matlab_io.h"
#if HAVE_MATIO

#include <stdexcept>
#include <boost/algorithm/string.hpp>

#include "multi_array_iterator.h"

using namespace boost;
using namespace std;

static const size_t N = data_t::dimensionality;

// Reader
matlab_reader::matlab_reader(std::string filename) : mat(nullptr), var(nullptr) {
	mat = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
	if(!mat)
		throw runtime_error("Couldn't open MAT file for reading.");
}

void matlab_reader::close() {
	if(mat) {
		expect0(Mat_Close(mat), "Couldn't close MAT file.");
		mat = nullptr;
	}
}

void matlab_reader::read() {
	if(var) return;
	while((var = Mat_VarReadNext(mat))) {
		if(var->rank > (int)data_t::dimensionality) {
			Mat_VarFree(var);
			continue; // Dimension too high.
		}
		switch(var->class_type) {
			case MAT_C_DOUBLE:
				break; // OK
			default:
				Mat_VarFree(var);
				continue; // Unsupported type.
		}
		return; // Accept this variable.
	}
}

matlab_reader::operator bool() {
	read();
	return var != nullptr;
}

void matlab_reader::operator>>(std::pair<std::string, data_t> &out) {
	read();
	if(!var)
		throw runtime_error("No more data available.");

	out.first = string(var->name);

	// fill inner dimensions first, others 1.
	std::array<data_t::size_type, N> extents;
	fill(extents.begin(), extents.end(), 1);
	copy_backward(&var->dims[0], &var->dims[var->rank], extents.end());

	// reference to the data
	multi_array_ref<double, N> data_ref(reinterpret_cast<double *>(var->data), extents);

	// copy data to output.
	out.second.resize(extents);
	out.second = data_ref;

	// delete data
	Mat_VarFree(var);
	var = nullptr;
}



// Writer
matlab_writer::matlab_writer(std::string filename) : mat(nullptr) {
	mat = Mat_Create(filename.c_str(), "SMRE " VERSION " (libmatio writer)");
	if(!mat)
		throw runtime_error("Couldn't create new MAT file.");
}

void matlab_writer::close() {
	if(mat) {
		expect0(Mat_Close(mat), "Couldn't close MAT file after writing.");
		mat = nullptr;
	}
}

void matlab_writer::operator<<(const std::pair<std::string, data_t> &in) {
	assert(mat);

	// copy data into a flat array
	double *data = new double[in.second.num_elements()];
	auto elements = all_elements(in.second);
	copy(elements.begin(), elements.end(), data);

	// strip unused dimensions
	int dims[data_t::dimensionality];
	size_t rank = 0;
	for(size_t i = 0 ; i < data_t::dimensionality ; i++) {
		const int d = in.second.shape()[i];
		if(d > 1) dims[rank++] = d;
	}


	// create variable header
	matvar_t *var = Mat_VarCreate(
		in.first.c_str(),
		MAT_C_DOUBLE, MAT_T_DOUBLE,
		rank, dims,
		data,
		MEM_CONSERVE
	);
	if(!var)
		throw runtime_error("Couldn't create variable.");

	expect0(Mat_VarWrite(mat, var, /*compress*/false), "Couldn't write variable.");

	Mat_VarFree(var);
	delete[] data;
}



// Factories
static bool is_mat(const string &filename) {
	return iends_with(filename, ".mat");
}

file_reader *matlab_io::reader(string filename) const {
	if(is_mat(filename))
		return new matlab_reader(filename);
	return nullptr;
}

file_writer *matlab_io::writer(string filename) const {
	if(is_mat(filename))
		return new matlab_writer(filename);
	return nullptr;
}



#endif
