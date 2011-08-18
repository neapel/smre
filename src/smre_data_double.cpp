#include "smre_data_double.h"
#include <iterator>

using namespace std;


void smre_data_double::print() {
	cout << *this << endl;
}


ostream &operator<<(ostream &o, smre_data_double &d) {
	o << d._name << " (";
	for(int i = 0 ; i < d.no_of_dims ; i++) {
		if(i > 0) o << 'x';
		o << d.dims[i];
	}
	o << ") = ";
	copy(d.data.begin(), d.data.end(), ostream_iterator<double>(o, " "));
	return o;
}
