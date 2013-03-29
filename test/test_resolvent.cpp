#include <gtkmm.h>
#include "resolvent.h"
#include "multi_array_operators.h"
#include "multi_array_io.h"
#include <algorithm>

int main(int argc, char **argv){
	std::cout << argv[1] << std::endl;

	if(argc != 2)
		throw std::invalid_argument("Missing input argument.");

	Gtk::Main kit(argc, argv);
	// read test image
	auto pb = Gdk::Pixbuf::create_from_file(argv[1]);
	boost::multi_array<float,2> rhs = pixbuf_to_multi_array(pb);

	const auto size = boost::extents_of(rhs);
	const auto width = size[0], height = size[1];

	// h1 resolvent
	h1resolvent my_h1resolvent(width, height, 1.0, 10.0);
	boost::multi_array<float, 2> sol = my_h1resolvent.evaluate(rhs);
	std::string file_name = "eval_h1resolvent.png";
	multi_array_to_pixbuf(sol)->save(file_name, "png");

	// l2 resolvent
	resolvent my_resolvent(10.0);
	sol = my_resolvent.evaluate(rhs);
	file_name = "eval_l2resolvent.png";
	multi_array_to_pixbuf(sol)->save(file_name, "png");

	return(0);
}
