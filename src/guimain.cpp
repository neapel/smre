#include <gtkmm.h>
#include <iostream>
#include <memory>
#include <boost/regex.hpp>
#include <stdexcept>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include "constraint_parser.h"
#include <boost/program_options.hpp>
#include <cmath>
#include "config.h"
#include "multi_array_io.h"
#include "chambolle_pock.h"


using namespace std;
using namespace Gtk;
using namespace Gdk;
using namespace Glib;
using namespace sigc;

typedef float T;


struct main_window : Gtk::ApplicationWindow {
	shared_ptr<params<T>> p;
	RefPtr<Pixbuf> input_image;

	SpinButton alpha_value{Adjustment::create(p->alpha, 0, 1, 0.1, 1), 0, 2};
	SpinButton tau_value{Adjustment::create(p->tau, 0, 10000, 10, 100), 0, 0};
	SpinButton sigma_value{Adjustment::create(p->sigma, 0, 5, 0.01, 0.1), 0.1, 4};
	SpinButton max_steps_value{Adjustment::create(p->max_steps, 1, 1000)};
	SpinButton force_q_value{Adjustment::create(p->force_q, 0, 100, 0.01, 0.1), 0, 3};
	SpinButton mc_steps_value{Adjustment::create(p->monte_carlo_steps, 1, 10000, 10, 100), 0, 0};
	CheckButton use_gpu_value{"Use OpenCL"};
	CheckButton use_fft_value{"Use FFT for convolution"};
	CheckButton resolv_value{"Use H₁ resolvent"};
	CheckButton penalized_scan_value{"Use penalized scan"};
	CheckButton debug_value{"Debug log"};
	CheckButton do_force_q_value{"Threshold (q)"};
	CheckButton auto_range_value{"Display with auto range"};
	Entry kernels_value;

	Statusbar statusbar;
	ProgressBar progress;
	Notebook notebook;
	Image input_image_view, output_image_view;

	struct cs : TreeModel::ColumnRecord {
		TreeModelColumn<RefPtr<Pixbuf>> img, old_img;
		TreeModelColumn<string> name;
		cs() { add(img); add(old_img); add(name); }
	} steps_columns;
	RefPtr<ListStore> steps_model{ListStore::create(steps_columns)};
	TreeView steps_view{steps_model};

	Menu constraints_menu;
	RefPtr<Action> load_image{Action::create("load_image", Stock::OPEN, "_Load")};
	RefPtr<Action> run{Action::create("run", Stock::MEDIA_PLAY, "_Run")};
	RefPtr<Action> stop{Action::create("stop", Stock::MEDIA_STOP, "_Stop")};

	struct debug_state {
		RefPtr<Pixbuf> img;
		string desc;
	};
	vector<debug_state> current_log, previous_log;
	RefPtr<Pixbuf> current_output;

	double progress_value;
	string progress_desc;

	Threads::Thread *current_thread = nullptr;
	bool continue_run = true;

	Dispatcher update_progress, update_output, algorithm_done;

	main_window(shared_ptr<params<T>> p, bool debug = false) : p(p) {
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
		options->set_row_spacing(5);
		int row = 0;

		options->attach(use_gpu_value, 0, row++, 2, 1);

		auto kernels_label = manage(new Label("Kernel sizes (h)", ALIGN_START));
		options->attach(*kernels_label, 0, row, 1, 1);
		kernels_value.set_placeholder_text("1,3,...,10; 50..55");
		if(p->kernel_sizes.empty()) kernels_value.set_text("2,4,...,50");
		else kernels_value.set_text(p->kernel_sizes.expr);
		options->attach(kernels_value, 1, row++, 1, 1);

		options->attach(use_fft_value, 0, row++, 2, 1);

		auto max_steps_label = manage(new Label("Max. steps", ALIGN_START));
		options->attach(*max_steps_label, 0, row, 1, 1);
		options->attach(max_steps_value, 1, row++, 1, 1);

		auto tau_label = manage(new Label("Step (τ)", ALIGN_START));
		options->attach(*tau_label, 0, row, 1, 1);
		options->attach(tau_value, 1, row++, 1, 1);

		auto sigma_label = manage(new Label("Step (σ)", ALIGN_START));
		options->attach(*sigma_label, 0, row, 1, 1);
		options->attach(sigma_value, 1, row++, 1, 1);

		options->attach(resolv_value, 0, row++, 2, 1);

		options->attach(do_force_q_value, 0, row, 1, 1);
		options->attach(force_q_value, 1, row++, 1, 1);

		auto mc_steps_label = manage(new Label("MC Steps", ALIGN_START));
		options->attach(*mc_steps_label, 0, row, 1, 1);
		options->attach(mc_steps_value, 1, row++, 1, 1);

		options->attach(penalized_scan_value, 0, row++, 2, 1);

		debug_value.set_active(debug);
		options->attach(debug_value, 0, row++, 2, 1);

		options->attach(auto_range_value, 0, row++, 2, 1);

		// Connect to model
		alpha_value.signal_value_changed().connect([=]{ p->alpha = alpha_value.get_value(); });
		tau_value.signal_value_changed().connect([=]{ p->tau = tau_value.get_value(); });
		sigma_value.signal_value_changed().connect([=]{ p->sigma = sigma_value.get_value(); });
		max_steps_value.signal_value_changed().connect([=]{ p->max_steps = max_steps_value.get_value(); });
		mc_steps_value.signal_value_changed().connect([=]{ p->monte_carlo_steps = mc_steps_value.get_value(); });

		auto q_ = [=]{
			const bool f = do_force_q_value.get_active();
			mc_steps_value.set_sensitive(!f);
			force_q_value.set_sensitive(f);
			p->force_q = f ? force_q_value.get_value() : -1;
		};
		do_force_q_value.set_active(p->force_q >= 0);
		do_force_q_value.signal_toggled().connect(q_);
		force_q_value.signal_value_changed().connect(q_);
		q_();

		use_gpu_value.set_active(p->use_gpu);
		use_gpu_value.signal_toggled().connect([=]{ p->use_gpu = use_gpu_value.get_active(); });

		use_fft_value.set_active(p->use_fft);
		use_fft_value.signal_toggled().connect([=]{ p->use_fft = use_fft_value.get_active(); });

		resolv_value.signal_toggled().connect([=]{
			if(resolv_value.get_active()) p->resolvent = make_shared<resolvent_h1_params<T>>();
			else p->resolvent = make_shared<resolvent_l2_params<T>>();
		});

		penalized_scan_value.set_active(p->penalized_scan);
		penalized_scan_value.signal_toggled().connect([=]{ p->penalized_scan = penalized_scan_value.get_active(); });

		kernels_value.signal_changed().connect([=]{
			p->kernel_sizes.clear();
			try {
				p->kernel_sizes = sizes_t(kernels_value.get_text());
				kernels_value.unset_icon(ENTRY_ICON_SECONDARY);
			} catch(invalid_argument e) {
				kernels_value.set_icon_from_stock(Stock::DIALOG_ERROR, ENTRY_ICON_SECONDARY);
			}
			validate();
		});

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
		progress.set_show_text(true);

		constraints_menu.show_all();
		progress.hide();
		show_all_children();
		set_default_size(1000, 800);

		update_progress.connect([&]{
			progress.set_fraction(progress_value);
			progress.set_text(progress_desc);
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

	void validate() {
		bool ok = true;
		ok &= !p->kernel_sizes.empty();
		ok &= input_image;
		run->set_sensitive(ok);
	}

	// Run the algorithm in a new thread.
	void do_run() {
		progress.show();
		run->set_sensitive(false);
		stop->set_sensitive(true);
		current_log.clear();

		// start thread
		continue_run = true;
		current_thread = Threads::Thread::create([=/* & doesn't work with threads */]{
			// The image to process
			auto input = pixbuf_to_multi_array(input_image);
			p->set_size(input.shape());
			auto run_p = p->runner();
			run_p->progress_cb = [=](double p, string d) {
				progress_value = p;
				progress_desc = d;
				update_progress();
			};
			run_p->current_cb = [=](const boost::multi_array<T,2> &a, size_t s) {
				current_output = multi_array_to_pixbuf(a, auto_range_value.get_active());
				update_output();
				return continue_run;
			};
			if(debug_value.get_active())
				run_p->debug_cb = [=](const boost::multi_array<T,2> &a, string desc) {
					current_log.push_back(debug_state{multi_array_to_pixbuf(a), desc});
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
		validate();
	}

	void open(string filename) {
		open(Pixbuf::create_from_file(filename));
	}

};


using namespace Gio;
using namespace boost::program_options;

struct app_t : Gtk::Application {
	bool debug = false;
	string input_file;
	shared_ptr<main_window> main;

	shared_ptr<params<T>> p = make_shared<params<T>>();
	shared_ptr<vex::Context> clctx;

	app_t()
	: Gtk::Application("smre.main", APPLICATION_HANDLES_COMMAND_LINE | APPLICATION_NON_UNIQUE) {}

	int on_command_line(const RefPtr<ApplicationCommandLine> &cmd) {
		try {
			clctx = make_shared<vex::Context>(vex::Filter::Count(1));
			vex::StaticContext<>::set(*clctx);
			cerr << "OpenCL: " << *clctx << endl;
			p->use_gpu = true;
			p->use_fft = false;
		} catch(cl::Error &e) {
			cerr << "OpenCL unavailable: " << e << endl;
			p->use_gpu = false;
		}

		string output_file;

		options_description main_desc("Options");
		main_desc.add_options()
			("help,h", "show this help")
			("debug,d", bool_switch(&debug), "dump intermediate steps")
			("input,i", value(&input_file)->value_name("<file>"),
				"input image file (png,jpeg,…)")
			("output,o", value(&output_file)->value_name("<file>"),
				"output image file (png) → disables GUI");
		if(p->use_gpu) main_desc.add_options()
			("cpu", value(&p->use_gpu)->implicit_value(false)->zero_tokens(),
				"Use CPU/FFTW (default: GPU/OpenCL)");
		main_desc.add_options()("fft", value(&p->use_fft)->implicit_value(true)->zero_tokens(),
			"Use FFT for convolution (default for CPU)");
		main_desc.add_options()("sat", value(&p->use_fft)->implicit_value(false)->zero_tokens(),
			"Use SAT for convolution (default for GPU)");
		options_description par_desc("Parameters");
		par_desc.add_options()
			("constraints,c", value(&p->kernel_sizes)->default_value(p->kernel_sizes)->value_name("<list>"),
				"List of kernel sizes, i.e. “1,7,4; 1,3,...,21; 2^2..8; 9” is a valid list")
			("resolvent,r", value(&p->resolvent)->default_value(p->resolvent)->value_name("<res>"),
				"Resolvent function to use, either “L2” for L₂ or “H1 <delta>” for H₁")
			("max-steps,#", value(&p->max_steps)->default_value(p->max_steps)->value_name("<int>"),
				"Maximum number of optimization steps")
			("tau,t", value(&p->tau)->default_value(p->tau)->value_name("<float>"),
				"Step size τ (large)")
			("sigma,s", value(&p->sigma)->default_value(p->sigma)->value_name("<float>"),
				"Step size σ (small)")
			("variance,v", value(&p->input_variance)->default_value(p->input_variance)->value_name("<float>"),
				"Set the input image variance explicitly instead of guessing it"); // TODO

		options_description q_desc("Threshold (q)");
		q_desc.add_options()
			("penalized-scan,p", bool_switch(&p->penalized_scan)->default_value(false),
				"Use penalized scan statistics")
			(",q", value(&p->force_q)->value_name("<float>"),
				"Set the threshold q to an explicit value instead of simulating it")
			("alpha,a", value(&p->alpha)->default_value(p->alpha)->value_name("<float>"),
				"Quantile α of q's distribution from simulation")
			("no-cache", bool_switch(&p->no_cache),
				"Don't use a cached value for q")
			("mc-steps", value(&p->monte_carlo_steps)->default_value(p->monte_carlo_steps)->value_name("<int>"),
				"Number of monte carlo simulations to use for q")
			("dump-mc", bool_switch(&p->dump_mc),
				"Dump all simulation data");

		options_description desc;
		desc.add(main_desc).add(par_desc).add(q_desc);
		int argc;
		char **argv = cmd->get_arguments(argc);
		variables_map vm;
		store(parse_command_line(argc, argv, desc), vm);
		notify(vm);
		if(vm.count("help")) {
			cerr << desc << endl;
			return EXIT_FAILURE;
		}

		// GUI
		if(output_file.empty()) {
			activate();
			return EXIT_SUCCESS;
		}

		// CLI
		if(input_file.empty()) {
			cerr << "--input required for --output." << endl;
			return EXIT_FAILURE;
		}

		auto input = pixbuf_to_multi_array(Pixbuf::create_from_file(input_file));
		p->set_size(input.shape());
		auto run_p = p->runner();
		run_p->progress_cb = [](double p, string d) {
			cerr << " " << int(p * 100) << "% " << d << "\r" << flush;
		};
		if(debug) {
			size_t i = 0;
			run_p->debug_cb = [&](const boost::multi_array<T,2> &a, string desc) {
				multi_array_to_pixbuf(a)->save(str(boost::format("%s.%d.%s.png") % output_file % (i++) % desc), "png");
			};
		}
		auto output = run_p->run(input);
		multi_array_to_pixbuf(output)->save(output_file, "png");

		return EXIT_SUCCESS;
	}

	void on_activate() {
		main = make_shared<main_window>(p, debug);
		add_window(*main);
		if(!input_file.empty()) main->open(input_file);
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
