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

struct labeled_image : VBox {
	Image image;
	Label label;
	labeled_image(RefPtr<Pixbuf> img, string text)
	: image(img), label(text) {
		pack_start(image, PACK_SHRINK);
		pack_start(label, PACK_SHRINK);
		show_all_children();
		show();
	}
	labeled_image(const boost::multi_array<float,2> &img, string text)
	: labeled_image(multi_array_to_pixbuf(img), text) {}
};

struct constraints_columns_t : TreeModel::ColumnRecord {
	TreeModelColumn<float> a;
	TreeModelColumn<float> b;
	TreeModelColumn<string> kernel;
	constraints_columns_t() {
		add(a); add(b); add(kernel);
	}
};

struct main_window : Gtk::Window {
	VBox vbox;
	Toolbar toolbar;
	Paned main;
	VBox left_pane;
	Grid options;
	Label tau_label{"τ", ALIGN_START};
	Entry tau_value;

	constraints_columns_t constraints_columns;
	RefPtr<ListStore> constraints_model;
	TreeView constraints_view;

	ScrolledWindow main_scroll;
	Viewport main_viewport;
	HBox images;

	Menu constraints_menu;
	RefPtr<Action> add_constraint{Action::create("add_constraint", Stock::ADD, "_Add Constraint")};
	RefPtr<Action> remove_constraint{Action::create("remove_constraint", Stock::REMOVE, "Remo_ve Constraint")};

	RefPtr<Action> load_image{Action::create("load_image", Stock::OPEN, "_Load Image")};
	RefPtr<Action> run{Action::create("run", Stock::EXECUTE, "_Run Chambolle-Pock")};

	Dispatcher algorithm_done;
	RefPtr<Pixbuf> input_image;
	vector<shared_ptr<labeled_image>> output_images;

	main_window()
	: constraints_model{ListStore::create(constraints_columns)},
	  constraints_view{constraints_model},
	  main_viewport{main_scroll.get_hadjustment(), main_scroll.get_vadjustment()}
	{
		// Main layout
		add(vbox);
		vbox.pack_start(toolbar, PACK_SHRINK);

		// Actions
		load_image->set_is_important(true);
		load_image->signal_activate().connect([&]{do_load_image();});
		toolbar.append(*load_image->create_tool_item());

		run->set_is_important(true);
		run->set_sensitive(false);
		run->signal_activate().connect([&]{do_run();});
		toolbar.append(*run->create_tool_item());

		// Main area
		vbox.pack_start(main);

		// Options
		main.pack1(left_pane, SHRINK);
		left_pane.pack_start(options, PACK_SHRINK);
		options.set_border_width(10);
		options.set_column_spacing(5);
		options.attach(tau_label, 0, 0, 1, 1);
		options.attach_next_to(tau_value, tau_label, POS_RIGHT, 1, 1);

		// Constraints Table
		add_constraint->signal_activate().connect([&]{
			auto row = *constraints_model->append();
			row[constraints_columns.a] = -1;
			row[constraints_columns.b] = 1;
			row[constraints_columns.kernel] = "box:1";
		});
		toolbar.append(*add_constraint->create_tool_item());
		constraints_menu.append(*add_constraint->create_menu_item());

		constraints_view.get_selection()->signal_changed().connect([&]{
			remove_constraint->set_sensitive(constraints_view.get_selection()->get_selected());
		});
		remove_constraint->signal_activate().connect([&]{
			auto it = constraints_view.get_selection()->get_selected();
			if(it) constraints_model->erase(it);
		});
		toolbar.append(*remove_constraint->create_tool_item());
		constraints_menu.append(*remove_constraint->create_menu_item());

		constraints_view.signal_button_press_event().connect_notify([&](GdkEventButton *evt){
			if(evt->type == GDK_BUTTON_PRESS && evt->button == GDK_BUTTON_SECONDARY)
				constraints_menu.popup(evt->button, evt->time);
		});

		left_pane.pack_start(constraints_view);
		constraints_view.append_column_editable("Kernel", constraints_columns.kernel);
		constraints_view.append_column_numeric_editable("a", constraints_columns.a, "%.2f");
		constraints_view.append_column_numeric_editable("b", constraints_columns.b, "%.2f");

		// Output
		main.pack2(main_scroll);
		main_scroll.add(main_viewport);
		main_viewport.add(images);

		constraints_menu.show_all();
		show_all_children();
		set_size_request(800, 600);

		algorithm_done.connect([&]{algorithm_done_f();});
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
			auto img = shared_ptr<labeled_image>(new labeled_image(input_image, "Input"));
			clear_images();
			output_images.push_back(img);
			images.pack_start(*img, PACK_SHRINK);
			run->set_sensitive(true);
		}
	}

	// Run the algorithm in a new thread.
	void do_run() {
		clear_images();
		Thread::create([&]{
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
			const float sigma = 0.5;
			const float gamma = 1;
			
			// run
			auto result = chambolle_pock(
				tau, sigma, gamma,
				input, constraints,
				[=](const boost::multi_array<float, 2> &x, string name){
					// collect intermediates
					output_images.push_back(shared_ptr<labeled_image>(new labeled_image(x, name)));
				}
			);
			output_images.push_back(shared_ptr<labeled_image>(new labeled_image(result, "Output")));
			algorithm_done(); // in main thread
		});
	}

	void algorithm_done_f() {
		// display intermediate images
		for(auto i : output_images)
			images.pack_start(*i, PACK_SHRINK);
		auto adj = main_scroll.get_hadjustment();
		adj->set_value(adj->get_upper());
	}

	void clear_images() {
		for(auto i : output_images)
			images.remove(*i);
		output_images.clear();
	}

};


int main(int argc, char **argv) {
	auto app = Application::create(argc, argv, "smre.main");
	main_window main;
	app->run(main);
}
