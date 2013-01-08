#ifndef __OPENCL_HELPERS_H__
#define __OPENCL_HELPERS_H__

#include "config.h"

#ifdef HAVE_OPENCL

#include <CL/cl.h>
#if HAVE_AMD_FFT
#	include <clAmdFft.h>
#endif
#include <string>
#include <fstream>
#include <iostream>
#include <array>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <boost/type_traits/is_same.hpp>
#include <boost/format.hpp>
#include <map>


/**
 * Simple C++ wrappers for OpenCL because <code>cl.hpp</code> is weird.
 */
namespace cl {

struct context;
struct buffer;
struct buffer_write;
struct buffer_read;
struct buffer_fill;
struct buffer_copy;
struct program;
struct kernel;
struct event;
struct after;
#if HAVE_AMD_FFT
struct fft;
struct fft_run;
#endif



/** Returns a string describing the status returned by an OpenCL call. */
std::string status_to_string(cl_int status);

/**
 * An OpenCL error, includes the status in its message.
 */
struct error : std::runtime_error {
	error(cl_int status, std::string message)
		: std::runtime_error(status_to_string(status) + ": " + message) {}
};



/**
 * An OpenCL device.
 */
struct device {
	cl_device_id native;
	device(cl_device_id o) : native(o) {}
};

/** Print human-readable information about the device. */
std::ostream &operator<<(std::ostream &, const device &);



/**
 * A platform to run OpenCL computations, consists of devices which can be used in groups.
 */
struct platform {
	cl_platform_id native;
	platform(cl_platform_id o) : native(o) {}

	/** Returns a list of all devices of the given type, for this platform. */
	std::vector<device> all_devices(cl_device_type type = CL_DEVICE_TYPE_ALL) const;
};

/** Print human-readable information about the platform. */
std::ostream &operator<<(std::ostream &, const platform &);

/** Return a list of all available platforms. */
std::vector<platform> all_platforms();



/**
 * A CL event, returned from asynchronous operations.
 * Wait for it using {@ref after::resume}.
 */
struct event {
	const context *parent_context;
	cl_event native;

	// DO allow copying

	event(const context *, cl_event);

	after then();
};

/** Output some information in human-readable format */
std::ostream &operator<<(std::ostream &, const event &);




/**
 * A list of events that have to occur before an action.
 */
struct after {
	const std::vector<event> events;
	const context *parent_context;

	// DON'T allow copying
	after(const after &) = delete;
	after(after &&a) : events(a.events), parent_context(a.parent_context) {};

	/**
	 * Construct the list from a list of events. This is the user constructor.
	 */
	after(std::initializer_list<event> events) : events(events), parent_context(nullptr) {
		for( event k : events ) {
			if( parent_context != nullptr && k.parent_context != parent_context )
				throw std::runtime_error("All events must be in the same context.");
			parent_context = k.parent_context;
		}
		if( parent_context == nullptr )
			throw std::runtime_error("No events in list.");
	}

	/** Internal constructor for empty lists */
	after(const context *ctx) : events(), parent_context(ctx) {}

	/** Queue a write operation after these events */
	event operator()(buffer_write &&);

	/** Queue a read operation after these events */
	event operator()(buffer_read &&);

	/** Queue a fill operation after these events */
	event operator()(buffer_fill &&);

	/** Queue a copy operation after these events */
	event operator()(buffer_copy &&);

	/** Queue a computation after these events */
	event operator()(const kernel &);

#if HAVE_AMD_FFT
	/** Queue an FFT after these events */
	event operator()(const fft_run &&r);
#endif

	/** Returns when all events were triggered. */
	void resume();

	/** Returns an event that is triggered when all events were triggered. */
	event barrier();
};



/**
 * The OpenCL context. Used for all access to the core.
 */
struct context {
	cl_context native;
	cl_device_id device;
	cl_command_queue queue;

	// DO allow copying
	context(const context &) = default;
	context(context &&) = default;

	~context();


	/** Create a context for the default platform */
	context();
	/** Create a context for a specific platform */
	explicit context(platform p);

	/** Load a program from a .cl code file. */
	program load(std::string filename);

	/** Load a program from a string. */
	program compile(std::string code);

	/** Enqueue an operation, see {@ref after::operator()} */
	template<typename T>
	event operator()(T &&op) {
		return after(this)( std::move(op) );
	}

	/** Wait for all operations to finish */
	void wait() {
		return after(this).resume();
	}
};

/** Output some information in human-readable format */
std::ostream &operator<<(std::ostream &, const context &);




/**
 * A program is a collection of kernels.
 */
struct program {
	cl_program native;
	context *parent;

	// DO allow copying although it's no use.
	program(const program &) = default;
	program(program &&) = default;


	program(context *parent, std::string code);

	/**
	 * Retrieve one kernel by its function name.
	 * Throws an {@ref error} if not found.
	 */
	kernel operator[](std::string name);
};

/** Output some information in human-readable format */
std::ostream &operator<<(std::ostream &, const program &);



/**
 * One kernel in a CL program.
 * Stores parameters and run configuration, too.
 */
struct kernel {
	cl_kernel native;
	context *parent_context;
	size_t argument_count;
	std::string name;
	std::vector<size_t> global_size;
	std::vector<size_t> global_offset;
	std::map<size_t, buffer *> buffer_arguments;

	// DO allow copying
	kernel(const kernel &) = default;
	kernel(kernel &&) = default;

	kernel(context *parent_context, cl_kernel k);

private:
	/** End of recursion. Checks if all arguments set */
	template<size_t i>
	void argsN() {
		if(i != argument_count)
			throw std::invalid_argument(boost::str(boost::format("%s Expected more arguments") % name));
	}

	/** Recursion: deconstructs argument list */
	template<size_t i, typename Head, typename... Tail>
	void argsN(Head &&head, Tail &&... tail) {
		arg(i, std::move(head));
		argsN<i + 1>(tail...);
	}

public:
	/** Set all arguments of the kernel to new values. */
	template<typename... T>
	kernel &operator()(T &&... a) {
		argsN<0u>(a...);
		return *this;
	}

	/**
	 * Set one positional argument of the kernel to a new value immediately.
	 */
	template<typename T>
	void arg(size_t i, const T &&data) {
		cl_uint status = clSetKernelArg(native, i, sizeof(T), &data);
		if(status != CL_SUCCESS)
			throw error(status, boost::str(boost::format("Couldn't set kernel %s argument %d to value") % name % i));
	}

	/**
	 * Set one positional argument of the kernel to a reference to the buffer.
	 * (defers until actual execution since buffer size might be unknown yet)
	 */
	void arg(size_t i, buffer &&buf) {
		buffer_arguments[i] = &buf;
	}

	/**
	 * Set the work size and offset (i.e. the range and offset for {@code get_global_id(dim)} in the kernel).
	 */
	inline kernel &size(std::vector<size_t> size, std::vector<size_t> offset = std::vector<size_t>()) {
		if(offset.size() != 0 && offset.size() != size.size())
			throw std::invalid_argument("Offset and size must have the same dimensions");
		global_size = size;
		global_offset = offset;
		return *this;
	}
};

/** Output some information in human-readable format */
std::ostream &operator<<(std::ostream &, const kernel &);



/**
 * Buffer access flags.
 */
enum buffer_access : int {
	/** Buffer is read from CPU */
	host_read = 1 << 1,
	/** Buffer is written to from CPU */
	host_write = 1 << 2,
	/** Buffer is read from GPU */
	kernel_read = 1 << 3,
	/** Buffer is written to from GPU */
	kernel_write = 1 << 4,
	/** Convenience: CPU to GPU */
	stream_in = host_write | kernel_read,
	/** Convenience: GPU to CPU */
	stream_out = host_read | kernel_write,
	/** Convenience: Full access */
	full_access = stream_in | stream_out,
	/** Convenience: GPU only */
	temp = kernel_read | kernel_write,
};



/**
 * A fill operation that repeats the given pattern
 */
struct buffer_fill {
	buffer &buf;
	const void *pattern;
	size_t pattern_size, offset, size;
	buffer_fill(const buffer_fill &) = delete;
	buffer_fill(buffer_fill &&) = default;
	buffer_fill(buffer &buf, const void *pattern, size_t pattern_size, size_t offset = 0, size_t size = 0)
		: buf(buf), pattern(pattern), pattern_size(pattern_size), offset(offset), size(size) {}
};

/**
 * A copy operation between buffers.
 * size=0 means min(src.size, dst.size)
 */
struct buffer_copy {
	buffer &src_buffer, &dst_buffer;
	size_t src_offset, dst_offset, size;
	buffer_copy(const buffer_copy &) = delete;
	buffer_copy(buffer_copy &&) = default;
	buffer_copy(buffer &src_buffer, buffer &dst_buffer, size_t src_offset = 0, size_t dst_offset = 0, size_t size = 0)
		: src_buffer(src_buffer), dst_buffer(dst_buffer), src_offset(src_offset), dst_offset(dst_offset), size(size) {}
};


/**
 * A buffer that might be accessed from the CPU and GPU
 */
struct buffer {
	cl_mem native;
	size_t size;
	const context *parent_context;
	buffer_access flags;
	std::string name;

	buffer(const buffer &) = delete;
	buffer(buffer &&) = default;

	/** Create a buffer.
	 * @param size
	 * 	Size in bytes. If not set here, it will be fixed with the first data upload. Must set for output buffers.
	 * @param flags
	 * 	Access flags. Default allows everything, slowest performance.
	 */
	buffer(size_t size = 0, buffer_access flags = full_access, std::string name = "")
		: native(0), size(size), parent_context(nullptr), flags(flags), name(name) {}

	buffer(buffer_access flags, std::string name = "")
		: native(0), size(0), parent_context(nullptr), flags(flags), name(name) {}


	/** Fill data from the array into the buffer */
	template<typename T>
	inline buffer_fill operator=(const T &a) {
		return std::move( buffer_fill(*this, &a, sizeof(T)) );
	}

	template<typename T>
	inline buffer_fill fill(const T &a, size_t size = 0, size_t offset = 0) {
		return std::move( buffer_fill(*this, &a, sizeof(T), offset, sizeof(T) * size));
	}

	/** Copy data from one buffer to another */
	inline buffer_copy operator=(buffer &other) {
		return std::move( buffer_copy(other, *this) );
	}

};

/** Output some info in human-readable format */
std::ostream &operator<<(std::ostream &, const buffer &);



/**
 * A write operation that can be enqueued using {@ref after::operator()}
 */
struct buffer_write {
	buffer &buf;
	const void *data;
	size_t size;

	buffer_write(const buffer_write &) = delete;
	buffer_write(buffer_write &&) = default;
	buffer_write(buffer &buf, const void *data, size_t size)
		: buf(buf), data(data), size(size) {}
};

/** Write data from the array to the buffer */
template<typename T, size_t n>
static inline buffer_write operator<<(buffer &b, const std::array<T, n> &a) {
	return std::move( buffer_write(b, &a[0], sizeof(T) * n) );
}



/**
 * A read operation that can be enqueued using {@ref after::operator()}
 */
struct buffer_read {
	buffer &buf;
	void *data;
	size_t size;
	buffer_read(const buffer_read &) = delete;
	buffer_read(buffer_read &&) = default;
	buffer_read(buffer &buf, void *data, size_t size)
		: buf(buf), data(data), size(size) {}
};

/** Read data from the buffer into the array */
template<typename T, size_t n>
static inline buffer_read operator>>(buffer &b, std::array<T, n> &a) {
	return std::move( buffer_read(b, &a[0], sizeof(T) * n) );
}

/** Read data from the buffer into the variable */
template<typename T>
static inline buffer_read operator>>(buffer &b, T &a) {
	return std::move( buffer_read(b, &a, sizeof(T)) );
}




#if HAVE_AMD_FFT
/**
 * An FFT plan that can be run on the CL device.
 */
struct fft {
	clAmdFftPlanHandle native;
	const context *parent_context;
	size_t lengths[3];
	clAmdFftDim dim;
	
	fft(const fft &) = delete;
	fft(fft &&) = default;

	fft(size_t x) : native(0), parent_context(nullptr), lengths{x}, dim{CLFFT_1D} {}
	fft(size_t x, size_t y) : native(0), parent_context(nullptr), lengths{x, y}, dim{CLFFT_2D} {}
	fft(size_t x, size_t y, size_t z) : native(0), parent_context(nullptr), lengths{x, y, z}, dim{CLFFT_3D} {}
	fft(const std::vector<size_t> &s) : native(0), parent_context(nullptr) {
		switch(s.size()) {
			case 1: dim = CLFFT_1D; break;
			case 2: dim = CLFFT_2D; break;
			case 3: dim = CLFFT_3D; break;
			default: throw std::runtime_error("Only supports 1-3D transformations.");
		}
		std::copy(s.begin(), s.end(), lengths);
	}
	~fft();

	/** Run a forward FFT. */
	fft_run forward(buffer &in, buffer &out);
	fft_run forward(buffer &inout);

	/** Run a backward FFT. */
	fft_run backward(buffer &in, buffer &out);
	fft_run backward(buffer &inout);
};

/**
 * An FFT operation that can be enqueued using {@ref after::operator()}
 */
struct fft_run {
	fft &that;
	clAmdFftDirection	dir;
	buffer &in, &out;

	fft_run(const fft_run &) = delete;
	fft_run(fft_run &&) = default;
	fft_run(fft &that, clAmdFftDirection dir, buffer &in, buffer &out) : that(that), dir(dir), in(in), out(out) {}
};
#endif




}; // cl::

#endif
#endif
