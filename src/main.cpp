#include "smre.h"
#include <boost/program_options.hpp>
#include <iostream>

using namespace std;



int main(int argc, char **argv) {
	using namespace boost::program_options;

	// declare options
	options_description desc("Recognized options");
	desc.add_options()
		// basic stuff
		("help,h", "show this message")
		("version,V", "show the software version")
		// global options
		("dummy,d", value<int>()->default_value(2), "test");

	options_description hidden("Hidden options");
	hidden.add_options()
		("input-files", value<vector<string>>(), "input files");

	positional_options_description pos;
	pos.add("input-files", -1);

	// parse
	variables_map vm;
	try {
		options_description all_desc; 
		all_desc.add(desc).add(hidden);
		auto parser = command_line_parser(argc, argv).options(all_desc).positional(pos);
		store(parser.run(), vm);
	} catch(exception &e) {
		cerr << "Error parsing command line: " << e.what() << '\n';
		cerr << desc << endl;
		return 1;
	}

	notify(vm);

	// process
	if(vm.count("help")) {
		cout << "Usage: " << argv[0] << " [options] files..." << endl;
		cout << desc;
		return 0;
	}

	if(vm.count("version")) {
		cout << VERSION << endl;
		return 0;
	}

	if(!vm.count("input-files")) {
		cerr << "No input files!" << endl;
		return 1;
	}


	// do things
	const auto files = vm["input-files"].as<vector<string>>();

	for(auto file : files)
		cout << "nothing to do with: " << file << endl;

	cout << vm["dummy"].as<int>() << endl;


	return 0;
}
