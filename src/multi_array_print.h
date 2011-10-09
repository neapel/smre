#ifndef __MULTI_ARRAY_PRINT_H__
#define __MULTI_ARRAY_PRINT_H__

#include <boost/multi_array.hpp>
#include <iostream>
#include <sstream>


/*
 * Some hack to print multi_arrays as a list of formatted matrices
 * (not intended to be machine readable, just for debugging small arrays)
 */


template<class A, size_t N>
struct print_multi_array {
	std::ostream &operator()(std::ostream &o, const A &a) {
		print_multi_array<decltype(a[0]), N - 1> next;
		for(int i = 0 ; i != static_cast<int>(a.size()) ; i++) {
			auto i_ = i + a.index_bases()[0];
			next(o, a[i_]);
			o << (a.num_dimensions() > 3 ? ", " : " ");
			o << "i_" << a.num_dimensions() << " = " << i_;
			if(i != static_cast<int>(a.size()) - 1) o << '\n';
		}
		return o;
	}
};


template<class A>
struct print_multi_array<A, 1> {
	std::ostream &operator()(std::ostream &o, const A &a) {
		o << '[';
		for(auto i = a.begin() ; i != a.end() ; i++) {
			if(i != a.begin()) o << ' ';
			o << *i;
		}
		o << ']';
		return o;
	}
};


template<class A>
struct print_multi_array<A, 2> {
	std::ostream &operator()(std::ostream &o, const A &a) {
		// get cell width, pre-render into strings
		std::vector<std::string> cells;
		cells.reserve(a.num_elements());
		size_t width = 0;
		for(auto r = a.begin() ; r != a.end() ; r++)
			for(auto c = r->begin() ; c != r->end() ; c++) {
				std::ostringstream s;
				s.copyfmt(o);
				s << *c;
				cells.push_back(s.str());
				width = std::max(width, s.str().size());
			}

		// output with padding
		auto cell = cells.begin();
		const auto rows = a.shape()[0], cols = a.shape()[1];
		for(size_t r = 0 ; r != rows ; r++) {
			o << (r == 0 ? "⎛" : (r == rows - 1 ? "⎝" : "⎜"));
			for(size_t c = 0 ; c != cols ; c++) {
				if(c != 0) o << ' ';
				o << std::string(width - cell->size(), ' ');
				o << *cell;
				cell++;
			}
			o << (r == 0 ? "⎞" : (r == rows - 1 ? "⎠" : "⎟"));
			if(r != rows - 1) o << '\n';
		}
		return o;
	}
};



// Used to extract the number of dimensions
// So we can use generic code above and not mention internal names.
template<class T, size_t N, template<class, size_t> class A>
std::ostream &operator<<(std::ostream &o, const A<T, N> &a) {
	return print_multi_array<A<T, N>, N>()(o, a);
}

template<class T, size_t N, class Alloc, template<class, size_t, class> class A>
std::ostream &operator<<(std::ostream &o, const A<T, N, Alloc> &a) {
	return print_multi_array<A<T, N, Alloc>, N>()(o, a);
}



#endif
