#include <gtkmm.h>
#include <iostream>
#include <memory>
#include <boost/program_options.hpp>
#include <boost/regex.hpp>
#include <stdexcept>
#include <boost/format.hpp>
#include "config.h"

#include "chambolle_pock.h"
#include "kernel_generator.h"

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
RefPtr<Pixbuf> multi_array_to_pixbuf(const boost::multi_array<float, 2> &a, bool mark_outliers = true) {
	auto h = a.shape()[0], w = a.shape()[1];
	auto pb = Pixbuf::create(COLORSPACE_RGB, false, 8, w, h);
	auto px = pb->get_pixels();
	auto stride = pb->get_rowstride();
	for(size_t y = 0 ; y != h ; y++)
		for(size_t x = 0 ; x != w ; x++) {
			int value = 255 * (a[y][x] + 1) / 2;
			if(!mark_outliers) value = max(0, min(255, value));
			auto p = px + (y * stride + 3 * x);
			if(value > 255) { p[0] = p[1] = 0; p[2] = 255; }
			else if(value < 0) { p[0] = 255; p[1] = p[2] = 0; }
			else { p[0] = p[1] = p[2] = value; }
		}
	return pb;
}


struct user_constraint : constraint {
	string expr;

	user_constraint(float a, float b, string e) : constraint{a, b}, expr(e) {}

	boost::multi_array<float, 2> get_k(const boost::multi_array<float, 2> &img) {
		auto h = img.shape()[0], w = img.shape()[1];
		return kernel_from_string(expr)(w, h);
	};
};


struct main_window : Gtk::ApplicationWindow {
	chambolle_pock &p;
	RefPtr<Pixbuf> input_image;

	SpinButton tau_value, gamma_value, sigma_value, max_steps_value;
	Statusbar statusbar;
	Spinner progress;
	Notebook notebook;
	Image original_image, output_image;

	struct cc : TreeModel::ColumnRecord {
		TreeModelColumn<float> a, b;
		TreeModelColumn<string> kernel;
		cc() { add(a); add(b); add(kernel); }
	} constraints_columns;
	RefPtr<ListStore> constraints_model;
	TreeView constraints_view;

	struct cs : TreeModel::ColumnRecord {
		TreeModelColumn<RefPtr<Pixbuf>> img;
		TreeModelColumn<string> name;
		TreeModelColumn<string> n, i;
		TreeModelColumn<string> tau, sigma, theta;
		cs() { add(img); add(name); add(n); add(i); add(tau); add(sigma); add(theta); }
	} steps_columns;
	RefPtr<TreeStore> steps_model;
	TreeView steps_view;

	Menu constraints_menu;
	RefPtr<Action> add_constraint, remove_constraint, load_image, run;

	Dispatcher algorithm_done;

	main_window(chambolle_pock &p)
	: p(p),
	  tau_value{Adjustment::create(p.tau, 0, 1000), 0, 2},
	  gamma_value{Adjustment::create(p.gamma, -5, 5), 0, 2},
	  sigma_value{Adjustment::create(p.sigma, 0, 5), 0, 2},
	  max_steps_value{Adjustment::create(p.max_steps, 1, 100)},
	  constraints_model{ListStore::create(constraints_columns)},
	  constraints_view{constraints_model},
	  steps_model{TreeStore::create(steps_columns)},
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
		auto paned = manage(new Paned());
		vbox->pack_start(*paned);

		// Options
		auto left_pane = manage(new VBox());
		paned->pack1(*left_pane, SHRINK);
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
		#define connect_value(val, var) val.signal_value_changed().connect([&]{var = val.get_value();})
		connect_value(tau_value, p.tau);
		connect_value(gamma_value, p.gamma);
		connect_value(sigma_value, p.sigma);
		connect_value(max_steps_value, p.max_steps);

		// Constraints Table
		for(auto cons : p.constraints) {
			auto row = *constraints_model->append();
			row[constraints_columns.a] = cons->a;
			row[constraints_columns.b] = cons->b;
			row[constraints_columns.kernel] = dynamic_pointer_cast<user_constraint>(cons)->expr;
		}
		add_constraint->signal_activate().connect([&]{
			auto row = *constraints_model->append();
			row[constraints_columns.a] = -1;
			row[constraints_columns.b] = 1;
			row[constraints_columns.kernel] = "box:1";
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
		constraints_view.append_column_numeric_editable("a", constraints_columns.a, "%.2f");
		constraints_view.append_column_numeric_editable("b", constraints_columns.b, "%.2f");

		// Output
		paned->pack2(notebook);
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
		steps_view.set_grid_lines(TREE_VIEW_GRID_LINES_VERTICAL);
		steps_view.append_column("Description", steps_columns.name);
		steps_view.append_column("n", steps_columns.n);
		steps_view.append_column("i", steps_columns.i);
		steps_view.append_column("τ", steps_columns.tau);
		steps_view.append_column("σ", steps_columns.sigma);
		steps_view.append_column("ϴ", steps_columns.theta);
		steps_view.append_column("Image", steps_columns.img);

		auto progress_b = manage(new ToolButton(progress));
		progress_b->set_expand(true);
		progress_b->set_sensitive(false);
		toolbar->append(*progress_b);

		constraints_menu.show_all();
		show_all_children();
		set_default_size(800, 600);

		progress.hide();

		algorithm_done.connect([&]{
			// re-attach modified model
			steps_view.set_model(steps_model);
			steps_view.expand_all();
			progress.stop();
			progress.hide();
			notebook.set_current_page(1); // output
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
		steps_model->clear();
		auto row = *steps_model->append();
		row[steps_columns.img] = input_image;
		row[steps_columns.name] = "Input";
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
		p.constraints.clear();
		for(TreeRow r : constraints_model->children()) {
			const auto a = r.get_value(constraints_columns.a);
			const auto b = r.get_value(constraints_columns.b);
			const auto ks = r.get_value(constraints_columns.kernel);
			p.constraints.push_back(shared_ptr<constraint>(new user_constraint{a, b, ks}));
		}

		// Debug mode?
		p.debug = notebook.get_current_page() == 2;
		// detach model for modification in thread
		if(p.debug) steps_view.unset_model();

		// start thread
		Threads::Thread::create([&]{
			// The image to process
			auto input = pixbuf_to_multi_array(input_image);

			// run
			auto run_p = p;
			auto result = run_p.run(input);
			output_image.set(multi_array_to_pixbuf(result));

			if(p.debug) {
				int previous_step = -1;
				steps_model->clear();
				auto step_row = steps_model->append();
				auto input_row = step_row;
				(*input_row)[steps_columns.img] = input_image;
				(*input_row)[steps_columns.name] = "Input";
				for(auto l : run_p.debug_log) {
					if(l.n != previous_step) {
						previous_step = l.n;
						step_row = steps_model->append(input_row->children());
						(*step_row)[steps_columns.name] = "Step";
						(*step_row)[steps_columns.n] = boost::lexical_cast<string>(l.n);
					}
					auto row = *steps_model->append(step_row->children());
					row[steps_columns.img] = multi_array_to_pixbuf(l.img);
					row[steps_columns.name] = l.name;
					row[steps_columns.n] = boost::lexical_cast<string>(l.n);
					if(l.i >= 0) row[steps_columns.i] = boost::lexical_cast<string>(l.i);
					if(l.tau != -1) row[steps_columns.tau] = boost::lexical_cast<string>(l.tau);
					if(l.sigma != -1) row[steps_columns.sigma] = boost::lexical_cast<string>(l.sigma);
					if(l.theta != -1) row[steps_columns.theta] = boost::lexical_cast<string>(l.theta);
				};
			}

			algorithm_done();
		});
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
	chambolle_pock p;

	app_t() : Gtk::Application("smre.main", APPLICATION_HANDLES_COMMAND_LINE | APPLICATION_HANDLES_OPEN | APPLICATION_NON_UNIQUE) {}

	bool parse_constraint(const ustring &, const ustring &value, bool has_value) {
		using namespace boost;
		if(!has_value) return false;
		static regex r("(?<kernel>[^,]+)"
							"(,(?<a>-?\\d*\\.?\\d*)"
							",(?<b>-?\\d*\\.?\\d*))?");
		smatch m;
		if(!regex_match(string(value), m, r)) return false;
		const float a = m["a"].matched ? lexical_cast<float>(m["a"]) : -1;
		const float b = m["b"].matched ? lexical_cast<float>(m["b"]) :  1;
		auto kernel = m["kernel"];
		p.constraints.push_back(std::shared_ptr<constraint>(new user_constraint{a, b, kernel}));
		return true;
	}

	int on_command_line(const RefPtr<ApplicationCommandLine> &cmd) {
		// define arguments
		OptionContext ctx("[FILE]");
		OptionGroup group("params", "default parameters", "longer");

		group.add_entry(entry("tau", "initial value for tau"), (double&)p.tau);
		group.add_entry(entry("sigma", "initial value for sigma"), (double&)p.sigma);
		group.add_entry(entry("gamma", "initial value for gamma"), (double&)p.gamma);
		group.add_entry(entry("steps", "number of iteration steps"), (int&)p.max_steps);

		group.add_entry(entry("constraint", "kernels 'box:SIZE[,A,B]' or 'gauss:SIGMA[,A,B]'"),
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
			auto run_p = p;
			auto output = run_p.run(input);
			multi_array_to_pixbuf(output, false)->save(output_file, "png");
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
