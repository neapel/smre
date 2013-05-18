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
	shared_ptr<params<T>> p;
	RefPtr<Pixbuf> input_image;

	SpinButton alpha_value, tau_value, sigma_value, max_steps_value, force_q_value;
	CheckButton impl_value, use_fft_value, resolv_value, penalized_scan_value, debug_value, do_force_q_value, auto_range_value;
	Statusbar statusbar;
	ProgressBar progress;
	Notebook notebook;
	Image input_image_view, output_image_view;

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
	RefPtr<Action> add_constraint, remove_constraint, load_image, run, stop;

	struct debug_state {
		RefPtr<Pixbuf> img;
		string desc;
	};
	vector<debug_state> current_log, previous_log;
	RefPtr<Pixbuf> current_output;
	double progress_value;

	Threads::Thread *current_thread = nullptr;
	bool continue_run = true;

	Dispatcher update_progress, update_output, algorithm_done;

	main_window(shared_ptr<params<T>> p)
	: p(p),
	  alpha_value{Adjustment::create(p->alpha, 0, 1), 0, 2},
	  tau_value{Adjustment::create(p->tau, 0, 10000), 0, 0},
	  sigma_value{Adjustment::create(p->sigma, 0, 5), 0, 4},
	  max_steps_value{Adjustment::create(p->max_steps, 1, 1000)},
	  force_q_value{Adjustment::create(p->force_q, 0, 100), 0, 3},
	  impl_value{"Use OpenCL"},
	  use_fft_value{"Use FFT"},
	  resolv_value{"Use H₁ resolvent"},
	  penalized_scan_value{"Use penalized scan"},
	  debug_value{"Debug log"},
	  do_force_q_value{"q"},
	  auto_range_value{"Display auto range"},
	  constraints_model{ListStore::create(constraints_columns)},
	  constraints_view{constraints_model},
	  steps_model{ListStore::create(steps_columns)},
	  steps_view{steps_model},
	  add_constraint{Action::create("add_constraint", Stock::ADD, "_Add Constraint")},
	  remove_constraint{Action::create("remove_constraint", Stock::REMOVE, "Remo_ve Constraint")},
	  load_image{Action::create("load_image", Stock::OPEN, "_Load")},
	  run{Action::create("run", Stock::EXECUTE, "_Run")},
	  stop{Action::create("stop", Stock::STOP, "_Stop")}
	{
		// Main layout
		auto vbox = manage(new VBox());
		add(*vbox);
		auto toolbar = manage(new Toolbar());
		toolbar->get_style_context()->add_class(GTK_STYLE_CLASS_PRIMARY_TOOLBAR);
		vbox->pack_start(*toolbar, PACK_SHRINK);

		// Actions
		load_image->set_is_important(true);
		load_image->signal_activate().connect([&]{do_load_image();});
		toolbar->append(*load_image->create_tool_item());

		run->set_is_important(true);
		run->set_sensitive(false);
		run->signal_activate().connect([&]{do_run();});
		toolbar->append(*run->create_tool_item());

		stop->set_is_important(true);
		stop->set_sensitive(false);
		stop->signal_activate().connect([&]{ continue_run = false; });
		toolbar->append(*stop->create_tool_item());

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
		options->attach_next_to(auto_range_value, debug_value, POS_BOTTOM, 2, 1);

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
			if(resolv_value.get_active()) p->resolvent = make_shared<resolvent_h1_params<T>>();
			else p->resolvent = make_shared<resolvent_l2_params<T>>();
		});

		penalized_scan_value.set_active(p->penalized_scan);
		penalized_scan_value.signal_toggled().connect([=]{ p->penalized_scan = penalized_scan_value.get_active(); });

		// Constraints Table
		for(auto cons : p->kernel_sizes) {
			auto row = *constraints_model->append();
			row[constraints_columns.kernel] = cons;
		}
		add_constraint->signal_activate().connect([&]{
			auto row = *constraints_model->append();
			row[constraints_columns.kernel] = 3;
		});
		constraints_menu.append(*add_constraint->create_menu_item());

		constraints_view.get_selection()->signal_changed().connect([&]{
			remove_constraint->set_sensitive(constraints_view.get_selection()->get_selected());
		});
		remove_constraint->signal_activate().connect([&]{
			auto it = constraints_view.get_selection()->get_selected();
			if(it) constraints_model->erase(it);
		});
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
		original_scroll->add(input_image_view);
		notebook.append_page(*original_scroll, "Input");
		auto output_scroll = manage(new ScrolledWindow(original_scroll->get_hadjustment(), original_scroll->get_vadjustment()));
		output_scroll->add(output_image_view);
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

		update_progress.connect([&]{
			progress.set_fraction(progress_value);
		});

		update_output.connect([&]{
			output_image_view.set(current_output);
			notebook.set_current_page(1);
		});

		algorithm_done.connect([&]{
			if(current_thread) current_thread->join();
			current_thread = nullptr;
			if(debug_value.get_active()) {
				steps_model->clear();
				for(size_t i = 0 ; i < current_log.size() ; i++) {
					auto row = *steps_model->append();
					row[steps_columns.name] = current_log[i].desc;
					row[steps_columns.img] = current_log[i].img;
					if(i < previous_log.size())
						row[steps_columns.old_img] = previous_log[i].img;
				}
				swap(current_log, previous_log);
				notebook.set_current_page(2);
			}
			progress.hide();
			run->set_sensitive(true);
			stop->set_sensitive(false);
		});
	}

	// Run the algorithm in a new thread.
	void do_run() {
		if(constraints_model->children().size() == 0) {
			cerr << "no constraints" << endl;
			return;
		}
		progress.show();
		run->set_sensitive(false);
		stop->set_sensitive(true);
		current_log.clear();

		p->kernel_sizes.clear();
		for(TreeRow r : constraints_model->children())
			p->kernel_sizes.push_back(r.get_value(constraints_columns.kernel));

		// start thread
		continue_run = true;
		current_thread = Threads::Thread::create([=/* & doesn't work with threads */]{
			// The image to process
			auto input = pixbuf_to_multi_array(input_image);
			p->set_size(input.shape());
			auto run_p = p->runner();
			run_p->progress_cb = [=](double p) {
				progress_value = p;
				update_progress();
			};
			run_p->current_cb = [=](const boost::multi_array<T,2> &a, size_t s) {
				current_output = multi_array_to_pixbuf(a, auto_range_value.get_active());
				update_output();
				return continue_run;
			};
			if(debug_value.get_active())
				run_p->debug_cb = [=](const boost::multi_array<T,2> &a, string desc) {
					#pragma omp critical
					{
						auto img = multi_array_to_pixbuf(a);
						current_log.push_back(debug_state{img, desc});
					}
				};
			auto result = run_p->run(input);
			force_q_value.set_value(run_p->q);
			algorithm_done();
		});
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
		input_image_view.set(input_image);
		output_image_view.set(input_image);
		run->set_sensitive(true);
	}

	void open(string filename) {
		open(Pixbuf::create_from_file(filename));
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
	shared_ptr<main_window> main;

	shared_ptr<params<T>> p = make_shared<params<T>>();
	shared_ptr<vex::Context> clctx;

	app_t()
	: Gtk::Application("smre.main", APPLICATION_HANDLES_COMMAND_LINE | APPLICATION_HANDLES_OPEN | APPLICATION_NON_UNIQUE),
	  input_image() {}

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
		if(value == "l2") p->resolvent = make_shared<resolvent_l2_params<T>>();
		else if(value == "h1") p->resolvent = make_shared<resolvent_h1_params<T>>();
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
		//group.add_entry(entry("debug", "enable debug output"), p->debug);
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
		clctx = make_shared<vex::Context>(vex::Filter::Count(1));
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
		main = make_shared<main_window>(p);
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
