#ifndef __MULTI_ARRAY_VECTOR_ADAPTOR_H__
#define __MULTI_ARRAY_VECTOR_ADAPTOR_H__

#include <algorithm>
#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/vector.hpp>

namespace boost { namespace numeric { namespace ublas {


// vector storage
template<class V>
class multi_array_vector_storage
 : public storage_array<multi_array_vector_storage<V>> {
	static_assert(V::dimensionality == 1, "Must be a rank 1 multi_array(view)");

	typedef multi_array_vector_storage<V> self_type;
	V &base;

public:
	// Assignable
	multi_array_vector_storage(const self_type &o) : base(o.base) {}

	// Container
	typedef typename V::element value_type;
	typedef typename V::iterator iterator;
	typedef typename V::const_iterator const_iterator;
	typedef typename V::reference reference;
	typedef typename V::const_reference const_reference;
	typedef value_type *pointer;
	typedef typename V::difference_type difference_type;
	typedef typename V::size_type size_type;

	iterator begin() { return base.begin(); }
	const_iterator begin() const { return base.begin(); }
	iterator end() { return base.end(); }
	const_iterator end() const { return base.end(); }
	size_type size() const { return base.size(); }
	size_type max_size() const { return base.size(); }
	bool empty() const { return base.num_elements() == 0; }
	void swap(self_type &other) { std::swap(base, other.base); }

	// Forward Container -> EqualityComparable, LessThanComparable
	bool operator==(const self_type &other) const { return base == other; }
	bool operator!=(const self_type &other) const { return base != other; }
	bool operator<(const self_type &other) const { return base < other; }
	bool operator>(const self_type &other) const { return base > other; }
	bool operator<=(const self_type &other) const { return base <= other; }
	bool operator>=(const self_type &other) const { return base >= other; }

	// Reversible Container
	typedef typename V::reverse_iterator reverse_iterator;
	typedef typename V::const_reverse_iterator const_reverse_iterator;

	reverse_iterator rbegin() { return base.rbegin(); }
	const_reverse_iterator rbegin() const { return base.rbegin(); }
	reverse_iterator rend() { return base.rend(); }
	const_reverse_iterator rend() const { return base.rend(); }

	// Random Access Container
	reference operator[](const size_type &idx) { return base[idx]; }
	const_reference operator[](const size_type &idx) const { return base[idx]; }

	// Storage
	multi_array_vector_storage(const size_type &size) : base(extents[size]) {}
	multi_array_vector_storage(const size_type &size, const value_type &value) : base(extents[size]) {
		fill(base.begin(), base.end(), value);
	}

	template<typename It>
	multi_array_vector_storage(It begin, It end) : base(extents[distance(begin, end)]) {
		base.assign(begin, end);
	}

	void resize(const size_type &new_size) {
		base.resize(extents[new_size]);
	}
	void resize(const size_type &new_size, const value_type &pad_value) {
		const auto old_size = base.size();
		resize(new_size);
		if(old_size < new_size)
			std::fill(base.begin() + old_size, base.end(), pad_value);
	}

	// Adapt multi_array
	multi_array_vector_storage(V &orig) : base(orig) {}
};



template<class T, template<class, size_t> class A>
multi_array_vector_storage<A<T, 1>> make_storage(A<T, 1> &a) {
	return {a};
}

template<class T, class Alloc, template<class, size_t, class> class A>
multi_array_vector_storage<A<T, 1, Alloc>> make_storage(A<T, 1, Alloc> &a) {
	return {a};
}


template<class A>
vector<typename A::value_type, multi_array_vector_storage<A>> make_vector(A &a) {
	auto s = make_storage(a);
	return {s.size(), s};
}





}}}

#endif
