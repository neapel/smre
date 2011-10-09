#ifndef __MULTI_ARRAY_ITERATOR_H__
#define __MULTI_ARRAY_ITERATOR_H__


#include <algorithm>
#include <array>
#include <boost/iterator.hpp>
#include <boost/multi_array.hpp>



namespace internal {
	template<class A>
	std::array<typename A::index, A::dimensionality>
	index_to_array(const A &a, typename A::index i) {
		std::array<typename A::index, A::dimensionality> idx;
		for(auto m = A::dimensionality ; m != 0 ; m--) {
			const auto n = m - 1;
			idx[n] = a.index_bases()[n] + (i % a.shape()[n]);
			i /= a.shape()[n];
		}
		assert(i == 0);
		return idx;
	}
}


// Iterates through all elements of a multi_array.
template<class A>
class element_iterator : public boost::iterator_facade<
	element_iterator<A>, typename A::element,
	std::random_access_iterator_tag
> {
	typedef element_iterator<A> self_type;

	A &base;
	typename A::index i;

public:
	element_iterator(A &a, typename A::index i) : base(a), i(i) {}
	element_iterator(A &a) : base(a), i(0) {}
	element_iterator(const self_type &o) : base(o.base), i(o.i) {}

	typename A::element &dereference() const {
		return base(internal::index_to_array(base, i));
	}

	bool equal(const self_type &other) const {
		return base == other.base && i == other.i;
	}

	ptrdiff_t distance_to(const self_type &other) const {
		return other.i - i;
	}

	void increment() { i++; }
	void decrement() { i--; }
	void advance(ptrdiff_t n) { i += n; }

	// Container
	self_type begin() const { return self_type(base, 0); }
	self_type end() const { return self_type(base, base.num_elements()); }
};







template<class A>
class const_element_iterator : public boost::iterator_facade<
	const_element_iterator<A>, const typename A::element,
	std::random_access_iterator_tag
> {
	typedef const_element_iterator<A> self_type;

	const A &base;
	typename A::index i;

public:
	const_element_iterator(const A &a, typename A::index i) : base(a), i(i) {}
	const_element_iterator(const A &a) : base(a), i(0) {}
	const_element_iterator(const self_type &o) : base(o.base), i(o.i) {}

	const typename A::element &dereference() const {
		return base(internal::index_to_array(base, i));
	}

	bool equal(const self_type &other) const {
		return base == other.base && i == other.i;
	}

	ptrdiff_t distance_to(const self_type &other) const {
		return other.i - i;
	}

	void increment() { i++; }
	void decrement() { i--; }
	void advance(ptrdiff_t n) { i += n; }

	// Container
	self_type begin() const { return self_type(base, 0); }
	self_type end() const { return self_type(base, base.num_elements()); }
};



template<class A>
element_iterator<A> all_elements(A &a) { return {a}; }

template<class A>
const_element_iterator<A> all_elements(const A &a) { return {a}; }


#endif
