#ifndef __MULTI_ARRAY_MATRIX_ADAPTOR_H__
#define __MULTI_ARRAY_MATRIX_ADAPTOR_H__

#include <boost/multi_array.hpp>
#include <boost/numeric/ublas/matrix.hpp>



namespace boost { namespace numeric { namespace ublas {




template<class M>
class multi_array_matrix
  : public matrix_container<multi_array_matrix<M>> {

	typedef multi_array_matrix<M> self_type;
	typedef M multi;

	multi base;
public:
#ifdef BOOST_UBLAS_ENABLE_PROXY_SHORTCUTS
	using matrix_container<self_type>::operator ();
#endif

	typedef typename multi::size_type size_type;
	typedef typename multi::difference_type difference_type;

	typedef typename multi::element value_type;
	typedef const value_type &const_reference;
	typedef value_type &reference;
	typedef const value_type *const_pointer;
	typedef value_type *pointer;

	// ?!
	//typedef const matrix_reference<const self_type> const_closure_type;
	//typedef matrix_reference<self_type> closure_type;

	// ?!?!
	//typedef int vector_temporary_type;	 // vector able to store all elements of c_matrix
	//typedef self_type matrix_temporary_type;

	// no dense_tag or we would have to provide data() as a model of Storage
	// sparse_tag probably assumes most entries are 0 which they probably aren't
	// so it's packed_tag.
	// I guess that prevents optimized algorithms from more effective linear index computations?
	typedef packed_tag storage_category;

	// well, iterator1 goes through rows?
	typedef row_major_tag orientation_category;


	// Construction and destruction
	multi_array_matrix() : base() {}

	// this breaks everything for some reason?!
	// multi_array_matrix(self_type &other) : base(other.base) {}

	// base may be a view, not constructable.
	//multi_array_matrix(size_type size1, size_type size2) : base(extents[size1][size2]) {}
	//template<class AE> multi_array_matrix(const matrix_expression<AE> &ae) : base() { matrix_assign<scalar_assign>(*this, ae); }

	explicit multi_array_matrix(multi base) : base(base) {}


	// Accessors
	size_type size1() const { return base.shape()[0]; }
	size_type size2() const { return base.shape()[1]; }
	
	const_pointer data() const { assert(false); return nullptr; }
	pointer data() { assert(false); return nullptr; }


	// Resizing. Preservation handled by multi_array, always assumed true.
	void resize(size_type size1, size_type size2, bool preserve = true) {
		base.resize(extents[size1][size2]);
	}


	// Element access. No access to base anywhere else, beware of nonzero origin.
	// uBLAS matrix always at zero origin.
	const_reference operator()(size_type i, size_type j) const {
		return base[i + base.index_bases()[0]][j + base.index_bases()[1]];
	}
	reference operator() (size_type i, size_type j) {
		return base[i + base.index_bases()[0]][j + base.index_bases()[1]];
	}

	reference at_element(size_type i, size_type j) { return operator()(i, j); }
	reference insert_element(size_type i, size_type j, const_reference t) { return operator()(i, j) = t; }
	

	// Zeroing
	void clear() {
		for(auto i = base.begin() ; i != base.end() ; i++)
			std::fill(i->begin(), i->end(), value_type());
	}


	// Assignment
#ifdef BOOST_UBLAS_MOVE_SEMANTICS
	/*! @note "pass by value" the key idea to enable move semantics */
	self_type &operator=(self_type m) {
		assign_temporary(m);
		return *this;
	}
#else
	self_type &operator=(const self_type &m) {
		base = m.base;
		return *this;
	}
#endif

	template<class C> // Container assignment without temporary
	self_type &operator=(const matrix_container<C> &m) {
		resize(m().size1(), m().size2(), false);
		assign(m);
		return *this;
	}
	
	self_type &assign_temporary(self_type &m) {
		swap(m);
		return *this;
	}

	template<class AE>
	self_type &operator=(const matrix_expression<AE> &ae) { 
		self_type temporary(ae);
		return assign_temporary(temporary);
	}

	template<class AE>
	self_type &assign(const matrix_expression<AE> &ae) { 
		matrix_assign<scalar_assign>(*this, ae); 
		return *this;
	}

	template<class AE>
	self_type &operator+=(const matrix_expression<AE> &ae) {
		self_type temporary(*this + ae);
		return assign_temporary(temporary);
	}

	template<class C> // Container assignment without temporary
	self_type &operator+=(const matrix_container<C> &m) { return plus_assign(m); }

	template<class AE>
	self_type &plus_assign(const matrix_expression<AE> &ae) { 
		matrix_assign<scalar_plus_assign>(*this, ae); 
		return *this;
	}

	template<class AE>
	self_type& operator-=(const matrix_expression<AE> &ae) {
		self_type temporary(*this - ae);
		return assign_temporary(temporary);
	}

	template<class C> // Container assignment without temporary
	self_type &operator-=(const matrix_container<C> &m) { return minus_assign(m); }

	template<class AE>
	self_type &minus_assign (const matrix_expression<AE> &ae) { 
		matrix_assign<scalar_minus_assign> (*this, ae); 
		return *this;
	}

	template<class AT>
	self_type& operator *= (const AT &at) {
		matrix_assign_scalar<scalar_multiplies_assign> (*this, at);
		return *this;
	}

	template<class AT>
	self_type& operator /= (const AT &at) {
		matrix_assign_scalar<scalar_divides_assign> (*this, at);
		return *this;
	}


	// Swapping
	void swap (self_type &m) { std::swap(base, m.base); }
	friend void swap (self_type &m1, self_type &m2) { m1.swap(m2); }


	// Iterator types
	// 1: columns -> rows
	// 2: rows -> columns
#ifdef BOOST_UBLAS_USE_INDEXED_ITERATOR
	typedef indexed_iterator1<self_type, dense_random_access_iterator_tag> iterator1;
	typedef indexed_iterator2<self_type, dense_random_access_iterator_tag> iterator2;
	typedef indexed_const_iterator1<self_type, dense_random_access_iterator_tag> const_iterator1;
	typedef indexed_const_iterator2<self_type, dense_random_access_iterator_tag> const_iterator2;
#else
	template<int R> class iterator;
	template<int R> class const_iterator;
	typedef iterator<0> iterator1;
	typedef iterator<1> iterator2;
	typedef const_iterator<0> const_iterator1;
	typedef const_iterator<1> const_iterator2;
#endif
	typedef reverse_iterator_base1<const_iterator1> const_reverse_iterator1;
	typedef reverse_iterator_base1<iterator1> reverse_iterator1;
	typedef reverse_iterator_base2<const_iterator2> const_reverse_iterator2;
	typedef reverse_iterator_base2<iterator2> reverse_iterator2;

	// Element lookup
private:
	// select a type by index. This is to enable the templated iterator below
	// Wasting some space to have less code duplication…
	template<class A, class B, int I> struct select_type {};
	template<class A, class B> struct select_type<A, B, 0> { typedef A type; };
	template<class A, class B> struct select_type<A, B, 1> { typedef B type; };

	template<int R>
	typename select_type<iterator1, iterator2, R>::type begin(size_type i = 0, size_type j = 0) {
		return typename select_type<iterator1, iterator2, R>::type(*this, R == 0 ? 0 : i, R == 1 ? 0 : j);
	}

	template<int R>
	typename select_type<const_iterator1, const_iterator2, R>::type begin(size_type i = 0, size_type j = 0) const {
		return typename select_type<const_iterator1, const_iterator2, R>::type(*this, R == 0 ? 0 : i, R == 1 ? 0 : j);
	}

	template<int R>
	typename select_type<iterator1, iterator2, R>::type end(size_type i = 0, size_type j = 0) {
		return typename select_type<iterator1, iterator2, R>::type(*this, R == 0 ? size1() : i, R == 1 ? size2() : j);
	}

	template<int R>
	typename select_type<const_iterator1, const_iterator2, R>::type end(size_type i = 0, size_type j = 0) const {
		return typename select_type<const_iterator1, const_iterator2, R>::type(*this, R == 0 ? size1() : i, R == 1 ? size2() : j);
	}

public:
	// Hard interface
	iterator1 begin1() { return begin<0>(); }
	const_iterator1 begin1() const { return begin<0>(); }

	iterator1 end1() { return end<0>(); }
	const_iterator1 end1() const { return end<0>(); }

	iterator2 begin2() { return begin<1>(0, 0); }
	const_iterator2 begin2() const { return begin<1>(0, 0); }

	iterator2 end2() { return end<1>(); }
	const_iterator2 end2() const { return end<1>(); }

	// Reverse iterators
	reverse_iterator1 rbegin1() { return reverse_iterator1(end1()); }
	const_reverse_iterator1 rbegin1() const { return const_reverse_iterator1(end1()); }
	reverse_iterator1 rend1() { return reverse_iterator1(begin1()); }
	const_reverse_iterator1 rend1() const { return const_reverse_iterator1(begin1()); }
	reverse_iterator2 rbegin2 () { return reverse_iterator2(end2()); }
	const_reverse_iterator2 rbegin2() const { return const_reverse_iterator2(end2()); }
	reverse_iterator2 rend2() { return reverse_iterator2(begin2()); }
	const_reverse_iterator2 rend2() const { return const_reverse_iterator2(begin2()); }


#ifndef BOOST_UBLAS_USE_INDEXED_ITERATOR
public:
	template<int R>
	class const_iterator
	: public container_const_reference<self_type>,
	  public random_access_iterator_base<dense_random_access_iterator_tag, const_iterator<R>, value_type> {
		friend class iterator<R>;
		size_type index[2];
		size_type linear() const { return index[0] * (*this)().size2() + index[1]; }

		typedef const_iterator<R> this_type;

	public:
		typedef typename self_type::difference_type difference_type;
		typedef typename self_type::value_type value_type;
		typedef typename self_type::const_reference reference;
		typedef typename self_type::const_pointer pointer;

		typedef typename select_type<const_iterator2, const_iterator1, R>::type dual_iterator_type;
		typedef typename select_type<const_reverse_iterator2, const_reverse_iterator1, R>::type dual_reverse_iterator_type;

		// Construction and destruction
		const_iterator() : container_const_reference<self_type>(), index{0,0} {}
		const_iterator(const self_type &m, size_type index1, size_type index2) : container_const_reference<self_type>(m), index{index1, index2} {}

		const_iterator(this_type &it) : container_const_reference<self_type>(it()), index{it.index1(), it.index2()} {}
		const_iterator(this_type &&it) : container_const_reference<self_type>(it()), index{it.index1(), it.index2()} {}
		explicit const_iterator(dual_iterator_type &it) : container_reference<self_type>(it()), index{it.index1(), it.index2()} {}
		explicit const_iterator(dual_iterator_type &&it) : container_reference<self_type>(it()), index{it.index1(), it.index2()} {}

		// Arithmetic
		this_type &operator++() { return operator+=(1); }
		this_type &operator--() { return operator-=(1); }
		this_type &operator+=(difference_type n) {
			index[R] += n;
			return *this;
		}
		this_type &operator-=(difference_type n) {
			index[R] -= n;
			return *this;
		}

		difference_type operator-(const this_type &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return index[R] - it.index[R];
		}

		// Dereference
		const_reference operator*() const { return (*this)()(index[0], index[1]); }	
		const_reference operator[](difference_type n) const {
			if(R == 0) return (*this)()(index[0] + n, index[1]);
			return (*this)()(index[0], index[1] + n);
		}
		
		dual_iterator_type begin() const { return (*this)().template begin<(R + 1) % 2>(index[0], index[1]); }
		dual_iterator_type end() const { return (*this)().template end<(R + 1) % 2>(index[0], index[1]); }
		
		dual_reverse_iterator_type rbegin() const { return dual_reverse_iterator_type(end()); }
		dual_reverse_iterator_type rend() const { return dual_reverse_iterator_type(begin()); }

		// Indices
		size_type index1() const { return index[0]; }
		size_type index2() const { return index[1]; }

		// Assignment
		this_type &operator=(const this_type &it) {
			container_const_reference<self_type>::assign(&it());
			index[0] = it.index[0];
			index[1] = it.index[1];
			return *this;
		}

		// Comparison
		bool operator==(const this_type &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return linear() == it.linear();
		}
		
		bool operator<(const this_type &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return index[R] < it.index[R];
		}
	};


	template<int R>
	class iterator
	: public container_reference<self_type>,
	  public random_access_iterator_base<dense_random_access_iterator_tag, iterator<R>, value_type> {
		friend class const_iterator<R>;
		size_type index[2];
		size_type linear() const { return index[0] * (*this)().size2() + index[1]; }

		typedef iterator<R> this_type;

	public:
		typedef typename self_type::difference_type difference_type;
		typedef typename self_type::value_type value_type;
		typedef typename self_type::reference reference;
		typedef typename self_type::pointer pointer;

		typedef typename select_type<iterator2, iterator1, R>::type dual_iterator_type;
		typedef typename select_type<reverse_iterator2, reverse_iterator1, R>::type dual_reverse_iterator_type;

		// Construction and destruction
		iterator() : container_reference<self_type>(), index{0,0} {}
		iterator(self_type &m, size_type index1, size_type index2) : container_reference<self_type>(m), index{index1, index2} {}

		iterator(this_type &it) : container_reference<self_type>(it()), index{it.index1(), it.index2()} {}
		iterator(this_type &&it) : container_reference<self_type>(it()), index{it.index1(), it.index2()} {}
		explicit iterator(dual_iterator_type &it) : container_reference<self_type>(it()), index{it.index1(), it.index2()} {}
		explicit iterator(dual_iterator_type &&it) : container_reference<self_type>(it()), index{it.index1(), it.index2()} {}

		// Arithmetic
		this_type &operator++() { return operator+=(1); }
		this_type &operator--() { return operator-=(1); }
		this_type &operator+=(difference_type n) {
			index[R] += n;
			return *this;
		}
		this_type &operator-=(difference_type n) {
			index[R] -= n;
			return *this;
		}

		difference_type operator-(const this_type &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return index[R] - it.index[R];
		}

		// Dereference
		reference operator*() { return (*this)()(index[0], index[1]); }	
		reference operator[](difference_type n) {
			if(R == 0) return (*this)()(index[0] + n, index[1]);
			return (*this)()(index[0], index[1] + n);
		}

		dual_iterator_type begin() { return (*this)().template begin<(R + 1) % 2>(index[0], index[1]); }
		dual_iterator_type end() { return (*this)().template end<(R + 1) % 2>(index[0], index[1]); }
		
		dual_reverse_iterator_type rbegin() { return dual_reverse_iterator_type(end()); }
		dual_reverse_iterator_type rend() { return dual_reverse_iterator_type(begin()); }

		// Indices
		size_type index1() const { return index[0]; }
		size_type index2() const { return index[1]; }

		// Assignment
		this_type &operator=(const this_type &it) {
			container_reference<self_type>::assign(&it());
			index[0] = it.index[0];
			index[1] = it.index[1];
			return *this;
		}

		// Comparison
		bool operator==(const this_type &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return linear() == it.linear();
		}
		
		bool operator<(const this_type &it) const {
			BOOST_UBLAS_CHECK(&(*this)() == &it(), external_logic());
			return index[R] < it.index[R];
		}
	};
#endif
};


#if 1
// Too restrictive, I want A[i] references and views, too.
template<class T>
multi_array_matrix<multi_array_ref<T, 2>> make_matrix(multi_array_ref<T, 2> &base) {
	return multi_array_matrix<multi_array_ref<T, 2>>(base);
}

// A[i] references:
template<class T>
multi_array_matrix<boost::detail::multi_array::sub_array<T, 2>> make_matrix(boost::detail::multi_array::sub_array<T, 2> &base) {
	return multi_array_matrix<boost::detail::multi_array::sub_array<T, 2>>(base);
}

// Views:
template<class T>
multi_array_matrix<boost::detail::multi_array::multi_array_view<T, 2>> make_matrix(boost::detail::multi_array::multi_array_view<T, 2> &base) {
	return multi_array_matrix<boost::detail::multi_array::multi_array_view<T, 2>>(base);
}
#else
// That's not a good idea:
template<class M>
multi_array_matrix<M> make_matrix(M base) {
	return multi_array_matrix<M>(base);
}
#endif




}}}




#endif
