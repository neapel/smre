#ifndef __SMRE_H__
#define __SMRE_H__

#include "config.h"
#include "debug_tools.h"

/*!
 * \mainpage
 * general description of everything...
 * \author foo bar
 */

#include <boost/multi_array.hpp>

/* The base array type.
 * Dimensions are from outer to inner:
 * Time, Z, Y, X */
typedef boost::multi_array<double, 4> data_t;


#endif
