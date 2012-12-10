#include <gtkmm.h>
#include <iostream>
#include <memory>
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
RefPtr<Pixbuf> multi_array_to_pixbuf(const boost::multi_array<float, 2> &a) {
	auto h = a.shape()[0], w = a.shape()[1];
	auto pb = Pixbuf::create(COLORSPACE_RGB, false, 8, w, h);
	auto px = pb->get_pixels();
	auto stride = pb->get_rowstride();
	for(size_t y = 0 ; y != h ; y++)
		for(size_t x = 0 ; x != w ; x++) {
			int value = 255 * (a[y][x] + 1) / 2;
			auto offset = y * stride + 3 * x;
			if(value > 255) {
				px[offset + 0] = 0;
				px[offset + 1] = 0;
				px[offset + 2] = 0xff;
			} else if(value < 0) {
				px[offset + 0] = 0xff;
				px[offset + 1] = 0;
				px[offset + 2] = 0;
			} else {
				for(size_t s = 0 ; s != 3 ; s++)
					px[offset + s] = value;
			}
		}
	return pb;
}


struct main_window : Gtk::Window {
	Entry tau_value;
	SpinButton gamma_value, sigma_value, max_steps_value;
	Statusbar statusbar;
	ProgressBar progress;
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
	RefPtr<Pixbuf> input_image;

	main_window()
	: gamma_value{Adjustment::create(1, -5, 5), 0, 2},
	  sigma_value{Adjustment::create(1, 0, 5), 0, 2},
	  max_steps_value{Adjustment::create(10, 1, 100)},
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
		tau_value.set_text("1000");
		auto gamma_label = manage(new Label("γ", ALIGN_START));
		options->attach_next_to(*gamma_label, *tau_label, POS_BOTTOM, 1, 1);
		options->attach_next_to(gamma_value, *gamma_label, POS_RIGHT, 1, 1);
		auto sigma_label = manage(new Label("σ", ALIGN_START));
		options->attach_next_to(*sigma_label, *gamma_label, POS_BOTTOM, 1, 1);
		options->attach_next_to(sigma_value, *sigma_label, POS_RIGHT, 1, 1);
		auto max_steps_label = manage(new Label("Steps", ALIGN_START));
		options->attach_next_to(*max_steps_label, *sigma_label, POS_BOTTOM, 1, 1);
		options->attach_next_to(max_steps_value, *max_steps_label, POS_RIGHT, 1, 1);

		// Constraints Table
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

		// Status
		vbox->pack_start(statusbar, PACK_SHRINK);
		statusbar.pack_end(progress);

		constraints_menu.show_all();
		show_all_children();
		//set_size_request(800, 600);

		progress.hide();

		algorithm_done.connect([&]{
			// re-attach modified model
			steps_view.set_model(steps_model);
			steps_view.expand_all();
			progress.hide();
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

		if(dialog.run() == RESPONSE_OK) {
			input_image = Pixbuf::create_from_file(dialog.get_filename());
			steps_model->clear();
			auto row = *steps_model->append();
			row[steps_columns.img] = input_image;
			row[steps_columns.name] = "Input";
			original_image.set(input_image);
			output_image.set(input_image);
			run->set_sensitive(true);
		}
	}

	// Run the algorithm in a new thread.
	void do_run() {
		// start thread
		Threads::Thread::create([&]{
			// Debug mode?
			const auto debug = notebook.get_current_page() == 2;
			progress.set_fraction(0);
			progress.show();
			// detach model for modification in thread
			if(debug) steps_view.unset_model();

			// The image to process
			auto input = pixbuf_to_multi_array(input_image);
			const auto h = input.shape()[0], w = input.shape()[1];

			// Actually create constraints now.
			vector<constraint> constraints;
			for(TreeRow r : constraints_model->children()) {
				const auto a = r.get_value(constraints_columns.a);
				const auto b = r.get_value(constraints_columns.b);
				const auto ks = r.get_value(constraints_columns.kernel);
				const auto kernel = kernel_from_string(ks)(w, h);
				constraints.push_back(constraint{a, b, input, kernel});
			}

			// Parameters
			const auto tau = boost::lexical_cast<float>(tau_value.get_text());
			const float sigma = sigma_value.get_value();
			const float gamma = gamma_value.get_value();
			const int max_steps = max_steps_value.get_value_as_int();

			TreeModel::iterator input_row, step_row;
			int previous_step = -1;
			if(debug) {
				steps_model->clear();
				step_row = input_row = steps_model->append();
				(*input_row)[steps_columns.img] = input_image;
				(*input_row)[steps_columns.name] = "Input";
			}
			
			// run
			auto result = chambolle_pock(
				max_steps,
				tau, sigma, gamma,
				input, constraints,
				[&](const boost::multi_array<float, 2> &x, string name, int n, int i, float tau, float sigma, float theta){
					if(debug) {
						if(n != previous_step) {
							previous_step = n;
							step_row = steps_model->append(input_row->children());
							(*step_row)[steps_columns.name] = "Step";
							(*step_row)[steps_columns.n] = boost::lexical_cast<string>(n);
						}
						auto row = *steps_model->append(step_row->children());
						row[steps_columns.img] = multi_array_to_pixbuf(x);
						row[steps_columns.name] = name;
						row[steps_columns.n] = boost::lexical_cast<string>(n);
						if(i >= 0) row[steps_columns.i] = boost::lexical_cast<string>(i);
						if(tau != -1) row[steps_columns.tau] = boost::lexical_cast<string>(tau);
						if(sigma != -1) row[steps_columns.sigma] = boost::lexical_cast<string>(sigma);
						if(theta != -1) row[steps_columns.theta] = boost::lexical_cast<string>(theta);
					}
					progress.set_fraction(1.0 * n / max_steps);
				}
			);
			output_image.set(multi_array_to_pixbuf(result));
			algorithm_done();
		});
	}

};


int main(int argc, char **argv) {
	auto app = Application::create(argc, argv, "smre.main");
	main_window main;
	app->run(main);
}
