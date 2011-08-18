
#include "../src/smre_data_double.h"

using namespace std;


int run(int no_of_runs) {
	int run, no_of_dims, no_of_elems, dim_no, *dims, elem_no, *ind;
	int max_no_of_dims;
	std::vector<double> data;

	max_no_of_dims = 3;

	cout << endl << endl << "************************************" << endl;
	cout << "TESTING smre_data_double " << endl;
	cout << "************************************" << endl;

	for (run = 0; run < no_of_runs; run++) {
		no_of_dims = (rand() % max_no_of_dims)+1;
		dims = new int[no_of_dims];
		ind = new int[no_of_dims];
		no_of_elems = 1;

		for (dim_no = 0; dim_no < no_of_dims; dim_no++) {
			dims[dim_no] = (rand() % 500) / no_of_dims;
			no_of_elems *= dims[dim_no];

			ind[dim_no] = 0;
		}

		for (;dim_no < max_no_of_dims; dim_no++)
			dims[dim_no] = 1;

		data.resize(no_of_elems, 0.0);
		for (elem_no = 0; elem_no < no_of_elems; elem_no++)
			data[elem_no] = elem_no;

		smre_data_double data_obj(data, dims, no_of_dims);

		if (data_obj.get_no_of_dimensions() != no_of_dims) {
			cout << "TEST OF smre_data_double WAS NOT SUCCESFUL!!!" << endl << endl;
			exit(-1);
		}

		int ind_0, ind_1, ind_2, test_op;

		int* check_dims = data_obj.get_dimensions();
		for (ind_0 = 0; ind_0 < no_of_dims; ind_0++) {
			if (check_dims[ind_0] != dims[ind_0]) {
				cout << "TEST OF smre_data_double WAS NOT SUCCESFUL!!!" << endl << endl;
				exit(-1);
			}
		}


		elem_no = 0;

		for (ind_2 = 0; ind_2 < dims[2]; ind_2++) {
			ind[2] = ind_2;
			for (ind_1 = 0; ind_1 < dims[1]; ind_1++) {
				ind[1] = ind_1;
				for (ind_0 = 0; ind_0 < dims[0]; ind_0++) {
					ind[0] = ind_0;
					test_op = data_obj(ind);
					if (test_op != elem_no) {
						cout << "TEST OF smre_data_double WAS NOT SUCCESFUL!!!" << endl << endl;
						exit(-1);
						return 1;
					}
					elem_no++;
				}
			}
		}




		delete dims;
		delete ind;
	}
	cout << endl << endl << "************************************" << endl;
	cout << "TEST OF smre_data_double SUCCESFULLY PASSED!!!" << endl;
	cout << "************************************" << endl;
	return 0;

};



int main(int, char **) {
	return run(100);
}
