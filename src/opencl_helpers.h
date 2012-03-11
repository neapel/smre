#ifndef __OPENCL_HELPERS_H__
#define __OPENCL_HELPERS_H__

#include <CL/cl.h>
#include <string>
#include <fstream>
#include <iostream>
#include <array>
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <boost/type_traits/is_same.hpp>

/**
 * Simple C++ wrappers for OpenCL because <code>cl.hpp</code> is weird.
 */
namespace cl {

struct context;
struct buffer;
struct buffer_write;
struct buffer_read;
struct program;
struct kernel;
struct event;
struct after;



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
	event(const event &) = default;
	event(event &&) = default;

	event(const context *, cl_event);
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
	after(after &&) = default;

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

	/** Queue a computation after these events */
	event operator()(const kernel &);

	/** Returns when all events were triggered. */
	void resume();
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
	std::vector<size_t> global_size;
	std::vector<size_t> global_offset;

	// DO allow copying
	kernel(const kernel &) = default;
	kernel(kernel &&) = default;

	kernel(context *parent_context, cl_kernel k);

private:
	/** End of recursion. Checks if all arguments set */
	template<size_t i>
	void argsN() {
		if(i != argument_count)
			throw std::invalid_argument("Expected more arguments");
	}

	/** Recursion: deconstructs argument list */
	template<size_t i, typename Head, typename... Tail>
	void argsN(Head &head, Tail &... tail) {
		arg(i, head);
		argsN<i + 1>(tail...);
	}

public:
	/** Set all arguments of the kernel to new values. */
	template<typename... T>
	kernel &args(T &... a) {
		argsN<0u>(a...);
		return *this;
	}

	/**
	 * Set one positional argument of the kernel to a new value.
	 */
	template<typename T>
	void arg(cl_uint i, const T &data) {
		cl_uint status = clSetKernelArg(
			native,
			i,
			sizeof(T),
			&data
		);
		if(status != CL_SUCCESS)
			throw error(status, "Couldn't set kernel argument to value");
	}

	/**
	 * Set one positional argument of the kernel to a reference to the buffer.
	 */
	void arg(cl_uint i, buffer &buf);


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







}; // cl::

#endif
