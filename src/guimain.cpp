#include <gtkmm.h>
#include <iostream>
#include <memory>
#include <boost/regex.hpp>
#include <stdexcept>
#include <boost/format.hpp>
#include "config.h"

#include "chambolle_pock.h"

using namespace std;
using namespace Gtk;
using namespace Gdk;
using namespace Glib;
using namespace sigc;

// Copy pixbuf data [0:255] into a new multi_array [-1:1]
boost::multi_array<float, 2> pixbuf_to_multi_array(const RefPtr<Pixbuf> &pb) {
	assert(pb->get_colorspace() == COLORSPACE_RGB);
	assert(pb->get_bits_per_sample() == 8);
	auto w = pb->get_width(), h = pb->get_height();
	auto px = pb->get_pixels();
	auto stride = pb->get_rowstride(), chans = pb->get_n_channels();
	boost::multi_array<float, 2> a(boost::extents[h][w]);
	for(int y = 0 ; y != h ; y++)
		for(int x = 0 ; x != w ; x++)
			a[y][x] = (px[y * stride + chans * x] / 255.0) * 2 - 1;
	return a;
}

// Copy multi_array data [-1:1] into a new pixbuf [0:255]
// mark_outliers: <-1 = red, >1 = blue, exactly 0 = green
RefPtr<Pixbuf> multi_array_to_pixbuf(const boost::multi_array<float, 2> &a) {
	auto h = a.shape()[0], w = a.shape()[1];
	auto pb = Pixbuf::create(COLORSPACE_RGB, false, 8, w, h);
	auto px = pb->get_pixels();
	auto stride = pb->get_rowstride();
	auto high = max(a), low = min(a);
	for(size_t y = 0 ; y != h ; y++)
		for(size_t x = 0 ; x != w ; x++) {
			auto p = px + (y * stride + 3 * x);
#if 1
			p[0] = p[1] = p[2] = high == low
				? 0
				: 255 * (a[y][x] - low) / (high - low);
#else
			auto f_val = a[y][x];
			if(mark_outliers && f_val == 0) { p[0] = p[2] = 0; p[1] = 200; }
			else {
				int value = 255 * (f_val + 1) / 2;
				if(!mark_outliers) value = max(0, min(255, value));
				if(value > 255) { p[0] = p[1] = 0; p[2] = 255; }
				else if(value < 0) { p[0] = 255; p[1] = p[2] = 0; }
				else { p[0] = p[1] = p[2] = value; }
			}
#endif
		}
	return pb;
}




struct main_window : Gtk::ApplicationWindow {
	chambolle_pock *const p;
	RefPtr<Pixbuf> input_image;

	SpinButton alpha_value, tau_value, gamma_value, sigma_value, max_steps_value;
	Statusbar statusbar;
	Spinner progress;
	Notebook notebook;
	Image original_image, output_image;

	struct cc : TreeModel::ColumnRecord {
		TreeModelColumn<string> kernel;
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
	RefPtr<ToggleAction> use_cl, use_debug;

	vector<debug_state> current_log, previous_log;
	Dispatcher algorithm_done;

	main_window(chambolle_pock *p)
	: p(p),
	  alpha_value{Adjustment::create(p->alpha, 0, 1), 0, 2},
	  tau_value{Adjustment::create(p->tau, 0, 1000), 0, 2},
	  gamma_value{Adjustment::create(p->gamma, -5, 5), 0, 2},
	  sigma_value{Adjustment::create(p->sigma, 0, 5), 0, 2},
	  max_steps_value{Adjustment::create(p->max_steps, 1, 100)},
	  constraints_model{ListStore::create(constraints_columns)},
	  constraints_view{constraints_model},
	  steps_model{ListStore::create(steps_columns)},
	  steps_view{steps_model},
	  add_constraint{Action::create("add_constraint", Stock::ADD, "_Add Constraint")},
	  remove_constraint{Action::create("remove_constraint", Stock::REMOVE, "Remo_ve Constraint")},
	  load_image{Action::create("load_image", Stock::OPEN, "_Load Image")},
	  run{Action::create("run", Stock::EXECUTE, "_Run Chambolle-Pock")},
	  use_cl{ToggleAction::create("use_cl", Stock::CONNECT, "Use OpenCL")},
	  use_debug{ToggleAction::create("use_cl", Stock::PROPERTIES, "Debug")}
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

		use_cl->set_is_important(true);
		toolbar->append(*use_cl->create_tool_item());

		use_debug->set_is_important(true);
		toolbar->append(*use_debug->create_tool_item());

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
		auto gamma_label = manage(new Label("γ", ALIGN_START));
		options->attach_next_to(*gamma_label, *tau_label, POS_BOTTOM, 1, 1);
		options->attach_next_to(gamma_value, *gamma_label, POS_RIGHT, 1, 1);
		auto sigma_label = manage(new Label("σ", ALIGN_START));
		options->attach_next_to(*sigma_label, *gamma_label, POS_BOTTOM, 1, 1);
		options->attach_next_to(sigma_value, *sigma_label, POS_RIGHT, 1, 1);
		auto max_steps_label = manage(new Label("Steps", ALIGN_START));
		options->attach_next_to(*max_steps_label, *sigma_label, POS_BOTTOM, 1, 1);
		options->attach_next_to(max_steps_value, *max_steps_label, POS_RIGHT, 1, 1);

		// Connect to model
		alpha_value.signal_value_changed().connect([=]{ p->alpha = alpha_value.get_value(); });
		tau_value.signal_value_changed().connect([=]{ p->tau = tau_value.get_value(); });
		gamma_value.signal_value_changed().connect([=]{ p->gamma = gamma_value.get_value(); });
		sigma_value.signal_value_changed().connect([=]{ p->sigma = sigma_value.get_value(); });
		max_steps_value.signal_value_changed().connect([=]{ p->max_steps = max_steps_value.get_value(); });

		use_cl->set_active(p->opencl);
		use_cl->signal_toggled().connect([=]{ p->opencl = use_cl->get_active(); });
		use_debug->set_active(p->opencl);
		use_debug->signal_toggled().connect([=]{ p->debug = use_debug->get_active(); });

		// Constraints Table
		for(auto cons : p->constraints) {
			auto row = *constraints_model->append();
			row[constraints_columns.kernel] = cons.expr;
		}
		add_constraint->signal_activate().connect([&]{
			auto row = *constraints_model->append();
			row[constraints_columns.kernel] = "box:3";
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

		left_pane->pack_start(constraints_view);
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
		p->constraints.clear();
		for(TreeRow r : constraints_model->children()) {
			const auto ks = r.get_value(constraints_columns.kernel);
			p->constraints.push_back(constraint(ks));
		}

		// start thread
		Threads::Thread::create([=/* & doesn't work with threads */]{
			// The image to process
			auto input = pixbuf_to_multi_array(input_image);

			// run
			auto run_p = *p;
			auto result = run_p.run(input);
			if(p->debug) {
				swap(current_log, previous_log);
				current_log = run_p.debug_log;
			}
			output_image.set(multi_array_to_pixbuf(result));

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
				if(current_log.size() == previous_log.size())
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

	// parameters.
	chambolle_pock *p;

	app_t() : Gtk::Application("smre.main", APPLICATION_HANDLES_COMMAND_LINE | APPLICATION_HANDLES_OPEN | APPLICATION_NON_UNIQUE), input_image(), main(NULL), p(new chambolle_pock()) {}

	bool parse_constraint(const ustring &, const ustring &value, bool has_value) {
		if(!has_value) return false;
		p->constraints.push_back(constraint{value});
		return true;
	}

	int on_command_line(const RefPtr<ApplicationCommandLine> &cmd) {
		// define arguments
		OptionContext ctx("[FILE]");
		OptionGroup group("params", "default parameters", "longer");

		group.add_entry(entry("alpha", "initial value for alpha"), p->alpha);
		group.add_entry(entry("tau", "initial value for tau"), p->tau);
		group.add_entry(entry("sigma", "initial value for sigma"), p->sigma);
		group.add_entry(entry("gamma", "initial value for gamma"), p->gamma);
		group.add_entry(entry("steps", "number of iteration steps"), p->max_steps);
		group.add_entry(entry("opencl", "use opencl implementation"), p->opencl);
		group.add_entry(entry("no-cache", "don't use the monte carlo cache"), p->no_cache);
		group.add_entry(entry("debug", "enable debug output"), p->debug);

		group.add_entry(entry("constraint", "kernels 'box:SIZE' or 'gauss:SIGMA'"),
			mem_fun(*this, &app_t::parse_constraint));

		string output_file;
		group.add_entry_filename(entry("output", "save output PNG here (runs without GUI)."), output_file);

		ctx.set_main_group(group);

		// parse them
		OptionGroup gtkgroup(gtk_get_option_group(true));
		ctx.add_group(gtkgroup);
		int argc;
		char **argv = cmd->get_arguments(argc);
		ctx.parse(argc, argv);

		// remaining arguments: filenames. open them.
		vector<RefPtr<File>> files;
		for(auto i = 1 ; i < argc ; i++)
			files.push_back(File::create_for_commandline_arg(argv[i]));
		if(!files.empty()) open(files);

		// run.
		if(output_file.empty())
			activate(); // show the gui.
		else if(input_image) {
			// CLI mode.
			auto input = pixbuf_to_multi_array(input_image);
			auto run_p = *p;
			auto output = run_p.run(input);
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
	return app_t().run(argc, argv);
}
