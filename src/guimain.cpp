#include <gtkmm.h>
#include <iostream>
#include <memory>
#include <boost/regex.hpp>
#include <stdexcept>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <cmath>
#include "config.h"
#include "multi_array_io.h"

#include "chambolle_pock.h"
#include "constraint_parser.h"

using namespace std;
using namespace Gtk;
using namespace Gdk;
using namespace Glib;
using namespace sigc;

typedef float T;


struct main_window : Gtk::ApplicationWindow {
	params<T> *const p;
	RefPtr<Pixbuf> input_image;

	SpinButton alpha_value, tau_value, sigma_value, max_steps_value, force_q_value;
	CheckButton impl_value, use_fft_value, resolv_value, penalized_scan_value, debug_value, do_force_q_value;
	Statusbar statusbar;
	Spinner progress;
	Notebook notebook;
	Image original_image, output_image;

	struct cc : TreeModel::ColumnRecord {
		TreeModelColumn<size_t> kernel;
		cc() { add(kernel); }
	} constraints_columns;
	RefPtr<ListStore> constraints_model;
	TreeView constraints_view;

	struct cs : TreeModel::ColumnRecord {
		TreeModelColumn<RefPtr<Pixbuf>> img, old_img;
		TreeModelColumn<string> name;
		cs() { add(img); add(old_img); add(name); }
	} steps_columns;
	RefPtr<ListStore> steps_model;
	TreeView steps_view;

	Menu constraints_menu;
	RefPtr<Action> add_constraint, remove_constraint, load_image, run;

	vector<debug_state<T>> current_log, previous_log;
	Dispatcher algorithm_done;

	main_window(params<T> *p)
	: p(p),
	  alpha_value{Adjustment::create(p->alpha, 0, 1), 0, 2},
	  tau_value{Adjustment::create(p->tau, 0, 1000), 0, 2},
	  sigma_value{Adjustment::create(p->sigma, 0, 5), 0, 2},
	  max_steps_value{Adjustment::create(p->max_steps, 1, 100)},
	  force_q_value{Adjustment::create(p->force_q, 0, 10), 0, 2},
	  impl_value{"Use OpenCL"},
	  use_fft_value{"Use FFT"},
	  resolv_value{"Use H1 resolvent"},
	  penalized_scan_value{"Use penalized scan"},
	  debug_value{"Debug log"},
	  do_force_q_value{"q"},
	  constraints_model{ListStore::create(constraints_columns)},
	  constraints_view{constraints_model},
	  steps_model{ListStore::create(steps_columns)},
	  steps_view{steps_model},
	  add_constraint{Action::create("add_constraint", Stock::ADD, "_Add Constraint")},
	  remove_constraint{Action::create("remove_constraint", Stock::REMOVE, "Remo_ve Constraint")},
	  load_image{Action::create("load_image", Stock::OPEN, "_Load Image")},
	  run{Action::create("run", Stock::EXECUTE, "_Run Chambolle-Pock")}
	{
		// Main layout
		auto vbox = manage(new VBox());
		add(*vbox);
		auto toolbar = manage(new Toolbar());
		vbox->pack_start(*toolbar, PACK_SHRINK);

		// Actions
		load_image->set_is_important(true);
		load_image->signal_activate().connect([&]{do_load_image();});
		toolbar->append(*load_image->create_tool_item());

		run->set_is_important(true);
		run->set_sensitive(false);
		run->signal_activate().connect([&]{do_run();});
		toolbar->append(*run->create_tool_item());

		// Main area
		auto paned = manage(new HBox());
		vbox->pack_start(*paned);

		// Options
		auto left_pane = manage(new VBox());
		paned->pack_start(*left_pane, PACK_SHRINK);
		auto options = manage(new Grid());
		left_pane->pack_start(*options, PACK_SHRINK);
		options->set_border_width(10);
		options->set_column_spacing(5);
		auto tau_label = manage(new Label("τ", ALIGN_START));
		options->attach(*tau_label, 0, 0, 1, 1);
		options->attach_next_to(tau_value, *tau_label, POS_RIGHT, 1, 1);
		auto sigma_label = manage(new Label("σ", ALIGN_START));
		options->attach_next_to(*sigma_label, *tau_label, POS_BOTTOM, 1, 1);
		options->attach_next_to(sigma_value, *sigma_label, POS_RIGHT, 1, 1);
		options->attach_next_to(do_force_q_value, *sigma_label, POS_BOTTOM, 1, 1);
		options->attach_next_to(force_q_value, do_force_q_value, POS_RIGHT, 1, 1);
		auto max_steps_label = manage(new Label("Steps", ALIGN_START));
		options->attach_next_to(*max_steps_label, do_force_q_value, POS_BOTTOM, 1, 1);
		options->attach_next_to(max_steps_value, *max_steps_label, POS_RIGHT, 1, 1);
		options->attach_next_to(impl_value, *max_steps_label, POS_BOTTOM, 2, 1);
		options->attach_next_to(use_fft_value, impl_value, POS_BOTTOM, 2, 1);
		options->attach_next_to(resolv_value, use_fft_value, POS_BOTTOM, 2, 1);
		options->attach_next_to(penalized_scan_value, resolv_value, POS_BOTTOM, 2, 1);
		options->attach_next_to(debug_value, penalized_scan_value, POS_BOTTOM, 2, 1);

		// Connect to model
		alpha_value.signal_value_changed().connect([=]{ p->alpha = alpha_value.get_value(); });
		tau_value.signal_value_changed().connect([=]{ p->tau = tau_value.get_value(); });
		sigma_value.signal_value_changed().connect([=]{ p->sigma = sigma_value.get_value(); });
		max_steps_value.signal_value_changed().connect([=]{ p->max_steps = max_steps_value.get_value(); });
		auto q_ = [=]{ p->force_q = do_force_q_value.get_active() ? force_q_value.get_value() : -1; };
		do_force_q_value.set_active(p->force_q >= 0);
		do_force_q_value.signal_toggled().connect(q_);
		force_q_value.signal_value_changed().connect(q_);

		impl_value.set_active(p->implementation == GPU_IMPL);
		impl_value.signal_toggled().connect([=]{ p->implementation = impl_value.get_active() ? GPU_IMPL : CPU_IMPL; });

		use_fft_value.set_active(p->use_fft);
		use_fft_value.signal_toggled().connect([=]{ p->use_fft = use_fft_value.get_active(); });

		resolv_value.signal_toggled().connect([=]{
			if(resolv_value.get_active()) p->resolvent = new resolvent_h1_params<T>();
			else p->resolvent = new resolvent_l2_params<T>();
		});

		penalized_scan_value.set_active(p->penalized_scan);
		penalized_scan_value.signal_toggled().connect([=]{ p->penalized_scan = penalized_scan_value.get_active(); });

		debug_value.set_active(p->debug);
		debug_value.signal_toggled().connect([=]{ p->debug = debug_value.get_active(); });



		// Constraints Table
		for(auto cons : p->kernel_sizes) {
			auto row = *constraints_model->append();
			row[constraints_columns.kernel] = cons;
		}
		add_constraint->signal_activate().connect([&]{
			auto row = *constraints_model->append();
			row[constraints_columns.kernel] = 3;
		});
		toolbar->append(*add_constraint->create_tool_item());
		constraints_menu.append(*add_constraint->create_menu_item());

		constraints_view.get_selection()->signal_changed().connect([&]{
			remove_constraint->set_sensitive(constraints_view.get_selection()->get_selected());
		});
		remove_constraint->signal_activate().connect([&]{
			auto it = constraints_view.get_selection()->get_selected();
			if(it) constraints_model->erase(it);
		});
		toolbar->append(*remove_constraint->create_tool_item());
		constraints_menu.append(*remove_constraint->create_menu_item());

		constraints_view.signal_button_press_event().connect_notify([&](GdkEventButton *evt){
			if(evt->type == GDK_BUTTON_PRESS && evt->button == GDK_BUTTON_SECONDARY)
				constraints_menu.popup(evt->button, evt->time);
		});

		auto constraints_scroll = manage(new ScrolledWindow());
		constraints_scroll->add(constraints_view);
		left_pane->pack_start(*constraints_scroll);
		constraints_view.append_column_editable("Kernel", constraints_columns.kernel);

		// Output
		paned->pack_start(notebook);
		// Scroll-locked image views for comparison
		auto original_scroll = manage(new ScrolledWindow());
		original_scroll->add(original_image);
		notebook.append_page(*original_scroll, "Input");
		auto output_scroll = manage(new ScrolledWindow(original_scroll->get_hadjustment(), original_scroll->get_vadjustment()));
		output_scroll->add(output_image);
		notebook.append_page(*output_scroll, "Output");
		// Debug details
		auto scrolled_window = manage(new ScrolledWindow());
		notebook.append_page(*scrolled_window, "Debug");
		scrolled_window->add(steps_view);
		steps_view.append_column("Description", steps_columns.name);
		steps_view.append_column("Image", steps_columns.img);
		steps_view.append_column("Previous", steps_columns.old_img);

		auto progress_b = manage(new ToolButton(progress));
		progress_b->set_expand(true);
		progress_b->set_sensitive(false);
		toolbar->append(*progress_b);

		constraints_menu.show_all();
		show_all_children();
		set_default_size(800, 600);

		progress.hide();

		algorithm_done.connect([&]{ on_algorithm_done(); });
	}

	// Select and load the image into a pixbuf.
	void do_load_image() {
		FileChooserDialog dialog(*this, "Please choose an image file", FILE_CHOOSER_ACTION_OPEN);
		dialog.add_button(Stock::CANCEL, RESPONSE_CANCEL);
		dialog.add_button(Stock::OPEN, RESPONSE_OK);
		auto filter = FileFilter::create();
		filter->set_name("Image files");
		filter->add_pixbuf_formats();
		dialog.add_filter(filter);

		if(dialog.run() == RESPONSE_OK)
			open(dialog.get_filename());
	}

	void open(RefPtr<Pixbuf> image) {
		input_image = image;
		original_image.set(input_image);
		output_image.set(input_image);
		run->set_sensitive(true);
	}

	void open(string filename) {
		open(Pixbuf::create_from_file(filename));
	}

	// Run the algorithm in a new thread.
	void do_run() {
		progress.start();
		progress.show();

		// Actually create constraints now.
		p->kernel_sizes.clear();
		for(TreeRow r : constraints_model->children()) {
			const auto ks = r.get_value(constraints_columns.kernel);
			p->kernel_sizes.push_back(ks);
		}

		// start thread
		Threads::Thread::create([=/* & doesn't work with threads */]{
			// The image to process
			auto input = pixbuf_to_multi_array(input_image);
			p->set_size(input.shape());
			auto run_p = p->runner();
			auto result = run_p->run(input);
			if(p->debug) {
				swap(current_log, previous_log);
				swap(current_log, run_p->debug_log);
			}
			output_image.set(multi_array_to_pixbuf(result));
			delete run_p;
			algorithm_done();
		});
	}

	void on_algorithm_done() {
		if(p->debug) {
			steps_model->clear();
			for(size_t i = 0 ; i < current_log.size() ; i++) {
				auto row = *steps_model->append();
				row[steps_columns.name] = current_log[i].name;
				row[steps_columns.img] = multi_array_to_pixbuf(current_log[i].img);
				if(i < previous_log.size())
					row[steps_columns.old_img] = multi_array_to_pixbuf(previous_log[i].img);
			}
		}
		progress.stop();
		progress.hide();
		notebook.set_current_page(p->debug ? 2 : 1); // output
	}

};


using namespace Gio;

OptionEntry entry(string name, string desc) {
	OptionEntry e;
	e.set_long_name(name);
	e.set_description(desc);
	return e;
}

struct app_t : Gtk::Application {
	RefPtr<Pixbuf> input_image;
	main_window *main;

	params<T> *p;
	vex::Context *clctx;


	app_t()
	: Gtk::Application("smre.main", APPLICATION_HANDLES_COMMAND_LINE | APPLICATION_HANDLES_OPEN | APPLICATION_NON_UNIQUE),
	  input_image(), main(NULL),
	  p(new params<T>()) {}

	bool parse_constraint(const ustring &, const ustring &value, bool has_value) {
		if(!has_value) return false;
		auto c = list_expression(value);
		copy(c.begin(), c.end(), back_inserter(p->kernel_sizes));
		return true;
	}

	bool parse_impl(const ustring &, const ustring &value, bool has_value) {
		if(!has_value) return false;
		if(value == "gpu") p->implementation = GPU_IMPL;
		else if(value == "cpu") p->implementation = CPU_IMPL;
		else return false;
		return true;	
	}

	bool parse_resolvent(const ustring &, const ustring &value, bool has_value) {
		if(!has_value) return false;
		if(value == "l2") p->resolvent = new resolvent_l2_params<T>();
		else if(value == "h1") p->resolvent = new resolvent_h1_params<T>();
		else return false;
		return true;
	}

	int on_command_line(const RefPtr<ApplicationCommandLine> &cmd) {
		// define arguments
		OptionContext ctx("[FILE]");
		OptionGroup group("params", "default parameters", "longer");

		double alpha = p->alpha, tau = p->tau, sigma = p->sigma, force_q = p->force_q;
		int max_steps = p->max_steps, monte_carlo_steps = p->monte_carlo_steps;
		bool use_fft = p->use_fft;
		group.add_entry(entry("alpha", "initial value for alpha"), alpha);
		group.add_entry(entry("tau", "initial value for tau"), tau);
		group.add_entry(entry("sigma", "initial value for sigma"), sigma);
		group.add_entry(entry("q", "don't calculate/cache q, use direct value"), force_q);
		group.add_entry(entry("steps", "number of iteration steps"), max_steps);
		group.add_entry(entry("impl", "'gpu' for OpenCL or 'cpu' for OpenMP."), mem_fun(*this, &app_t::parse_impl));
		group.add_entry(entry("mc-steps", "number of Monte Carlo steps"), monte_carlo_steps);
		group.add_entry(entry("no-cache", "don't use the monte carlo cache"), p->no_cache);
		group.add_entry(entry("penalized-scan", "use penalized scan statistic"), p->penalized_scan);
		group.add_entry(entry("mc-dump", "dump mc data for each kernel"), p->dump_mc);
		group.add_entry(entry("debug", "enable debug output"), p->debug);
		group.add_entry(entry("constraint", "kernel sizes, comma separated list or '<start>,<next>,...,<end>'"), mem_fun(*this, &app_t::parse_constraint));
		group.add_entry(entry("resolvent", "'l2' or 'h1'"), mem_fun(*this, &app_t::parse_resolvent));
		group.add_entry(entry("use-fft", "use FFT for convolution (or SAT)"), use_fft);

		string output_file;
		group.add_entry_filename(entry("output", "save output PNG here (runs without GUI)."), output_file);

		ctx.set_main_group(group);

		// parse them
		OptionGroup gtkgroup(gtk_get_option_group(true));
		ctx.add_group(gtkgroup);
		int argc;
		char **argv = cmd->get_arguments(argc);
		ctx.parse(argc, argv);
		p->alpha = alpha;
		p->tau = tau;
		p->sigma = sigma;
		p->force_q = force_q;
		p->max_steps = max_steps;
		p->monte_carlo_steps = monte_carlo_steps;
		p->use_fft = use_fft;

		// remaining arguments: filenames. open them.
		vector<RefPtr<File>> files;
		for(auto i = 1 ; i < argc ; i++)
			files.push_back(File::create_for_commandline_arg(argv[i]));
		if(!files.empty()) open(files);

		// init OpenCL
		clctx = new vex::Context(vex::Filter::Count(1));
		vex::StaticContext<>::set(*clctx);
		cerr << "CL context:" << *clctx << endl;

		// run.
		if(output_file.empty())
			activate(); // show the gui.
		else if(input_image) {
			// CLI mode.
			auto input = pixbuf_to_multi_array(input_image);
			p->set_size(input.shape());
			auto run_p = p->runner();
			auto output = run_p->run(input);
			multi_array_to_pixbuf(output)->save(output_file, "png");
		} else {
			cmd->printerr("When using --output, you must specify an input image, too.");
			return EXIT_FAILURE;
		}

		return EXIT_SUCCESS;
	}

	void on_open(const vector<RefPtr<File>> &files, const ustring &) {
		if(files.size() != 1) throw runtime_error("can only open one file.");
		input_image = Pixbuf::create_from_stream(files[0]->read());
	}

	void on_activate() {
		main = new main_window(p);
		add_window(*main);
		if(input_image) main->open(input_image);
		main->show();
	}
};

int main(int argc, char **argv) {
	try {
		return app_t().run(argc, argv);
	} catch(cl::Error &e) {
		cerr << e << endl;
	}
}
