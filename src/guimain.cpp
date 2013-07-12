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
	RefPtr<Pixbuf> input_image, output_image;
	SpinButton q_value{Adjustment::create(p->force_q, 0, 100, 0.1, 1), 0, 3};
	SpinButton stddev_value{Adjustment::create(1, 0, 10, 0.1, 1), 0, 2};

	CheckButton debug_value{"Record debug log"};
	CheckButton live_display_value{"Show output after each step"};
	CheckButton auto_range_value{"Display with adjusted range"};

	ProgressBar progress;
	Label progress_text;
	Notebook notebook;
	Image input_image_view, output_image_view;
	int zoom_level{1};
	RefPtr<Adjustment> hadjustment{Adjustment::create(0,0,0)};
	RefPtr<Adjustment> vadjustment{Adjustment::create(0,0,0)};

	struct cs : TreeModel::ColumnRecord {
		TreeModelColumn<RefPtr<Pixbuf>> img, old_img;
		TreeModelColumn<string> name;
		cs() { add(img); add(old_img); add(name); }
	} steps_columns;
	RefPtr<ListStore> steps_model{ListStore::create(steps_columns)};
	TreeView steps_view{steps_model};

	RefPtr<Action> load_image{Action::create("load_image", Stock::OPEN, "_Load image")};
	RefPtr<Action> save_image{Action::create("save_image", Stock::SAVE, "_Save image")};
	RefPtr<Action> run{Action::create("run", Stock::MEDIA_PLAY, "_Run")};
	RefPtr<Action> stop{Action::create("stop", Stock::MEDIA_STOP, "S_top")};
	RefPtr<ToggleAction> auto_run{ToggleAction::create("auto-run", Stock::REFRESH, "_Auto Run")};
	RefPtr<Action> zoom_in{Action::create("zoom_in", Stock::ZOOM_IN, "Zoom _In")};
	RefPtr<Action> zoom_out{Action::create("zoom_out", Stock::ZOOM_OUT, "Zoom _Out")};

	struct debug_state {
		RefPtr<Pixbuf> img;
		string desc;
	};
	vector<debug_state> current_log, previous_log;

	double progress_value;
	string progress_desc;

	Threads::Thread *current_thread = nullptr;
	Threads::Mutex mutex;
	bool continue_run = true, valid = false, have_run = false;

	Dispatcher update_progress, update_output, algorithm_done;

	main_window(shared_ptr<params<T>> p, bool debug, bool gpu_enabled) : p(p) {
		set_title("SMRE");

		// Main layout
		auto vbox = manage(new VBox());
		add(*vbox);
		auto toolbar = manage(new Toolbar());
		toolbar->get_style_context()->add_class(GTK_STYLE_CLASS_PRIMARY_TOOLBAR);
		vbox->pack_start(*toolbar, PACK_SHRINK);

		// Actions
		{
			load_image->set_is_important(true);
			toolbar->append(*load_image->create_tool_item());
			load_image->signal_activate().connect([=]{
				FileChooserDialog dialog(*this, "Please choose an image file", FILE_CHOOSER_ACTION_OPEN);
				dialog.add_button(Stock::CANCEL, RESPONSE_CANCEL);
				dialog.add_button(Stock::OPEN, RESPONSE_OK);
				auto filter = FileFilter::create();
				filter->set_name("Image files");
				filter->add_pixbuf_formats();
				dialog.add_filter(filter);
				if(dialog.run() == RESPONSE_OK)
					open(dialog.get_filename());
			});

			save_image->set_is_important(true);
			save_image->set_sensitive(false);
			toolbar->append(*save_image->create_tool_item());
			save_image->signal_activate().connect([=]{
				FileChooserDialog dialog(*this, "Please choose a filename to save as", FILE_CHOOSER_ACTION_SAVE);
				dialog.set_do_overwrite_confirmation();
				dialog.set_create_folders();
				dialog.add_button(Stock::CANCEL, RESPONSE_CANCEL);
				dialog.add_button(Stock::SAVE, RESPONSE_OK);
				auto filter = FileFilter::create();
				filter->set_name("PNG files");
				filter->add_pattern("*.png");
				dialog.add_filter(filter);
				if(dialog.run() == RESPONSE_OK) {
					output_image->save(dialog.get_filename(), "png");
				}
			});

			run->set_is_important(true);
			run->set_sensitive(false);
			toolbar->append(*run->create_tool_item());
			run->signal_activate().connect([&]{do_run();});

			stop->set_is_important(true);
			stop->set_sensitive(false);
			toolbar->append(*stop->create_tool_item());
			stop->signal_activate().connect([=]{
				continue_run = false;
				stop->set_sensitive(false);
			});

			auto_run->set_is_important(true);
			toolbar->append(*auto_run->create_tool_item());
			auto_run->signal_toggled().connect([=]{
				validate();
			});

			toolbar->append(*zoom_in->create_tool_item());
			zoom_in->signal_activate().connect([=]{
				zoom_level++;
				render();
			});
			toolbar->append(*zoom_out->create_tool_item());
			zoom_out->signal_activate().connect([=]{
				zoom_level--;
				render();
			});
		}

		// Main area
		auto paned = manage(new HBox());
		vbox->pack_start(*paned);

		// Options
		auto left_pane = manage(new VBox());
		left_pane->set_border_width(10);
		paned->pack_start(*left_pane, PACK_SHRINK);
		auto options = manage(new Grid());
		left_pane->pack_start(*options, PACK_SHRINK);
		options->set_column_spacing(5);
		options->set_row_spacing(5);
		int row = 0;

		// Kernels
		{
			auto kernels_label = manage(new Label("Box sizes (<i>h</i>)", ALIGN_START));
			kernels_label->set_use_markup();
			options->attach(*kernels_label, 0, row, 1, 1);
			auto kernels_value = manage(new Entry());
			kernels_value->set_placeholder_text("1,3,...,10; 50..55");
			if(p->kernel_sizes.empty()) kernels_value->set_text("2,4,...,50");
			else kernels_value->set_text(p->kernel_sizes.expr);
			options->attach(*kernels_value, 1, row++, 1, 1);

			auto k_ = [=]{
				p->kernel_sizes.clear();
				try {
					p->kernel_sizes = sizes_t(kernels_value->get_text());
					kernels_value->unset_icon(ENTRY_ICON_SECONDARY);
				} catch(invalid_argument e) {
					kernels_value->set_icon_from_stock(Stock::DIALOG_ERROR, ENTRY_ICON_SECONDARY);
				}
				validate();
			};
			kernels_value->signal_changed().connect(k_);
			k_();
		}

		// Implementation
		{
			auto impl_label = manage(new Label("Convolver", ALIGN_START));
			options->attach(*impl_label, 0, row, 1, 1);
			auto impl_box = manage(new HBox());
			impl_box->get_style_context()->add_class(GTK_STYLE_CLASS_LINKED);
			options->attach(*impl_box, 1, row++, 1, 1);
			RadioButton::Group impl_group;
			auto use_sat = manage(new RadioButton{impl_group, "SAT"});
			use_sat->set_sensitive(gpu_enabled);
			use_sat->set_mode(false);
			impl_box->pack_start(*use_sat);
			auto use_clfft = manage(new RadioButton{impl_group, "CLFFT"});
			use_clfft->set_sensitive(gpu_enabled);
			use_clfft->set_mode(false);
			impl_box->pack_start(*use_clfft);
			auto use_fftw = manage(new RadioButton{impl_group, "FFTW"});
			use_fftw->set_mode(false);
			impl_box->pack_start(*use_fftw);

			if(!p->use_gpu) use_fftw->set_active(true);
			else if(p->use_fft) use_clfft->set_active(true);
			else use_sat->set_active(true);
			auto impl_cb = [=]{
				if(use_fftw->get_active()) {
					p->use_gpu = false; p->use_fft = true;
				} else if(use_clfft->get_active()) {
					p->use_gpu = true; p->use_fft = true;
				} else {
					p->use_gpu = true; p->use_fft = false;
				}
				validate();
			};
			use_fftw->signal_toggled().connect(impl_cb);
			use_clfft->signal_toggled().connect(impl_cb);
			use_sat->signal_toggled().connect(impl_cb);
			impl_cb();
		}

		options->attach(*manage(new HSeparator), 0, row++, 2, 1);

		// Stop condition
		{
			auto max_steps_label = manage(new Label("Max steps", ALIGN_START));
			options->attach(*max_steps_label, 0, row, 1, 1);
			auto max_steps_value = manage(new SpinButton{Adjustment::create(p->max_steps, 1, 10000)});
			options->attach(*max_steps_value, 1, row++, 1, 1);
			max_steps_value->signal_value_changed().connect([=]{
				p->max_steps = max_steps_value->get_value();
				validate();
			});

			auto tolerance_label = manage(new Label("Tolerance⁻¹", ALIGN_START));
			tolerance_label->set_use_markup();
			options->attach(*tolerance_label, 0, row, 1, 1);
			auto tolerance_value = manage(new SpinButton{Adjustment::create(p->tolerance, 0, 100000, 10, 100), 100, 0});
			options->attach(*tolerance_value, 1, row++, 1, 1);
			tolerance_value->signal_value_changed().connect([=]{
				p->tolerance = tolerance_value->get_value();
				validate();
			});
		}

		// Step sizes
		{
			auto tau_label = manage(new Label("Step (τ₀)", ALIGN_START));
			options->attach(*tau_label, 0, row, 1, 1);
			auto tau_value = manage(new SpinButton{Adjustment::create(p->tau, 0, 10000, 10, 100), 0, 0});
			options->attach(*tau_value, 1, row++, 1, 1);
			tau_value->signal_value_changed().connect([=]{
				p->tau = tau_value->get_value();
				validate();
			});

			auto sigma_label = manage(new Label("Step (σ₀)", ALIGN_START));
			options->attach(*sigma_label, 0, row, 1, 1);
			auto sigma_value = manage(new SpinButton{Adjustment::create(p->sigma, 0, 5, 0.01, 0.1), 0.1, 4});
			options->attach(*sigma_value, 1, row++, 1, 1);
			sigma_value->signal_value_changed().connect([=]{
				p->sigma = sigma_value->get_value();
				validate();
			});
		}

		options->attach(*manage(new HSeparator), 0, row++, 2, 1);

		// Variance
		{
			auto method_label = manage(new Label("Std. dev. (σ)", ALIGN_START));
			options->attach(*method_label, 0, row, 1, 1);
			auto stddev_box = manage(new HBox());
			stddev_box->get_style_context()->add_class(GTK_STYLE_CLASS_LINKED);
			options->attach(*stddev_box, 1, row++, 1, 1);
			RadioButton::Group stddev_group;
			auto stddev_fixed = manage(new RadioButton{stddev_group, "fixed"});
			stddev_fixed->set_mode(false);
			stddev_box->pack_start(*stddev_fixed);
			auto stddev_mad = manage(new RadioButton{stddev_group, "median"});
			stddev_mad->set_mode(false);
			stddev_box->pack_start(*stddev_mad);

			auto stddev_label = manage(new Label("Value", ALIGN_START));
			options->attach(*stddev_label, 0, row, 1, 1);
			options->attach(stddev_value, 1, row++, 1, 1);

			stddev_value.set_value(p->input_stddev);
			if(p->input_stddev >= 0) stddev_fixed->set_active(true);
			else stddev_mad->set_active(true);

			auto stddev_cb = [=]{
				if(stddev_fixed->get_active()) {
					p->input_stddev = stddev_value.get_value();
					stddev_value.set_sensitive(true);
				} else {
					p->input_stddev = -1;
					stddev_value.set_sensitive(false);
				}
				validate();
			};
			stddev_fixed->signal_toggled().connect(stddev_cb);
			stddev_mad->signal_toggled().connect(stddev_cb);
			stddev_value.signal_value_changed().connect(stddev_cb);
			stddev_cb();
		}

		options->attach(*manage(new HSeparator), 0, row++, 2, 1);

		// Resolvent
		{
			auto resolv_label = manage(new Label("Resolvent", ALIGN_START));
			options->attach(*resolv_label, 0, row, 1, 1);
			auto resolv_box = manage(new HBox());
			resolv_box->get_style_context()->add_class(GTK_STYLE_CLASS_LINKED);
			options->attach(*resolv_box, 1, row++, 1, 1);
			RadioButton::Group res_group;
			auto resolv_h1 = manage(new RadioButton{res_group, "H¹"});
			resolv_h1->set_mode(false);
			resolv_box->pack_start(*resolv_h1);
			auto resolv_l2 = manage(new RadioButton{res_group, "L²"});
			resolv_l2->set_mode(false);
			resolv_box->pack_start(*resolv_l2);

			auto delta_label = manage(new Label("L²-mix (δ)", ALIGN_START));
			options->attach(*delta_label, 0, row, 1, 1);
			auto delta_value = manage(new SpinButton{Adjustment::create(0.5, 0, 1, 0.1, 1), 0, 2});
			options->attach(*delta_value, 1, row++, 1, 1);

			auto try_h1 = dynamic_pointer_cast<resolvent_h1_params<T>>(p->resolvent);
			if(try_h1) {
				delta_value->set_value(try_h1->delta);
				resolv_h1->set_active(true);
			} else {
				resolv_l2->set_active(true);
			}

			auto resolv_cb = [=]{
				if(resolv_h1->get_active()) {
					p->resolvent = make_shared<resolvent_h1_params<T>>(delta_value->get_value());
					delta_value->set_sensitive(true);
				} else {
					p->resolvent = make_shared<resolvent_l2_params<T>>();
					delta_value->set_sensitive(false);
				}
				validate();
			};
			resolv_h1->signal_toggled().connect(resolv_cb);
			resolv_l2->signal_toggled().connect(resolv_cb);
			delta_value->signal_value_changed().connect(resolv_cb);
			resolv_cb();
		}

		options->attach(*manage(new HSeparator), 0, row++, 2, 1);

		// q
		{
			auto method_label = manage(new Label("Threshold (<i>q</i>)", ALIGN_START));
			method_label->set_use_markup();
			options->attach(*method_label, 0, row, 1, 1);
			auto method_box = manage(new HBox());
			method_box->get_style_context()->add_class(GTK_STYLE_CLASS_LINKED);
			options->attach(*method_box, 1, row++, 1, 1);
			RadioButton::Group method_group;
			auto force_q = manage(new RadioButton{method_group, "fixed"});
			force_q->set_mode(false);
			method_box->pack_start(*force_q);
			auto simulate_q = manage(new RadioButton{method_group, "simulate"});
			simulate_q->set_mode(false);
			method_box->pack_start(*simulate_q);
			if(p->force_q >= 0) force_q->set_active(true);
			else simulate_q->set_active(true);

			auto q_label = manage(new Label("Value", ALIGN_START));
			options->attach(*q_label, 0, row, 1, 1);
			options->attach(q_value, 1, row++, 1, 1);
			q_value.signal_value_changed().connect([=]{
				if(!force_q->get_active()) return;
				p->force_q = q_value.get_value();
				validate();
			});

			auto alpha_label = manage(new Label("Quantile (α)", ALIGN_START));
			options->attach(*alpha_label, 0, row, 1, 1);
			auto alpha_value = manage(new SpinButton{Adjustment::create(p->alpha, 0, 1, 0.1, 1), 0, 2});
			options->attach(*alpha_value, 1, row++, 1, 1);
			alpha_value->signal_value_changed().connect([=]{
				p->alpha = alpha_value->get_value();
				validate();
			});

			auto mc_steps_label = manage(new Label("Steps", ALIGN_START));
			options->attach(*mc_steps_label, 0, row, 1, 1);
			auto mc_steps_value = manage(new SpinButton{Adjustment::create(p->monte_carlo_steps, 1, 10000, 10, 100), 0, 0});
			options->attach(*mc_steps_value, 1, row++, 1, 1);
			mc_steps_value->signal_value_changed().connect([=]{
				p->monte_carlo_steps = mc_steps_value->get_value();
				validate();
			});

			auto corr_label = manage(new Label("Correction", ALIGN_START));
			options->attach(*corr_label, 0, row, 1, 1);
			auto corr_box = manage(new HBox());
			corr_box->get_style_context()->add_class(GTK_STYLE_CLASS_LINKED);
			options->attach(*corr_box, 1, row++, 1, 1);
			RadioButton::Group corr_group;
			auto no_corr_value = manage(new RadioButton{corr_group, "none"});
			no_corr_value->set_mode(false);
			corr_box->pack_start(*no_corr_value);
			auto penalized_scan_value = manage(new RadioButton{corr_group, "penalized scan"});
			penalized_scan_value->set_mode(false);
			corr_box->pack_start(*penalized_scan_value);
			if(p->penalized_scan) penalized_scan_value->set_active(true);
			else no_corr_value->set_active(true);
			penalized_scan_value->signal_toggled().connect([=]{
				p->penalized_scan = penalized_scan_value->get_active();
				validate();
			});

			auto q_cb = [=]{
				const bool f = force_q->get_active();
				mc_steps_value->set_sensitive(!f);
				alpha_value->set_sensitive(!f);
				q_value.set_sensitive(f);
				p->force_q = f ? q_value.get_value() : -1;
				validate();
			};
			force_q->signal_toggled().connect(q_cb);
			q_cb();
		}

		options->attach(*manage(new HSeparator), 0, row++, 2, 1);

		// Misc
		{
			debug_value.set_active(debug);
			options->attach(debug_value, 0, row++, 2, 1);
			live_display_value.set_active(true);
			options->attach(live_display_value, 0, row++, 2, 1);
			options->attach(auto_range_value, 0, row++, 2, 1);
		}

		// Output
		notebook.set_margin_top(5);
		auto nb_ctx = notebook.get_style_context();
		auto notebook_bg = nb_ctx->get_background_color();
		output_image_view.override_background_color(notebook_bg);
		input_image_view.override_background_color(notebook_bg);
		steps_view.override_background_color(notebook_bg);
		paned->pack_start(notebook);
		// Scroll-locked image views for comparison
		auto original_scroll = manage(new ScrolledWindow(hadjustment, vadjustment));
		original_scroll->add(input_image_view);
		notebook.append_page(*original_scroll, "Input");
		auto output_scroll = manage(new ScrolledWindow(hadjustment, vadjustment));
		output_scroll->add(output_image_view);
		notebook.append_page(*output_scroll, "Output");
		// Debug details
		auto scrolled_window = manage(new ScrolledWindow());
		notebook.append_page(*scrolled_window, "Debug");
		scrolled_window->add(steps_view);
		steps_view.append_column("Description", steps_columns.name);
		steps_view.append_column("Image", steps_columns.img);
		steps_view.append_column("Previous", steps_columns.old_img);

		// Visible done.
		show_all_children();
		set_default_size(1000, 800);

		// Progress
		left_pane->pack_end(progress, PACK_SHRINK);
		left_pane->pack_end(progress_text, PACK_SHRINK);

		update_progress.connect([&]{
			Threads::Mutex::Lock lock(mutex);
			progress.set_fraction(progress_value);
			progress_text.set_text(progress_desc);
		});

		update_output.connect([&]{
			Threads::Mutex::Lock lock(mutex);
			render();
		});

		algorithm_done.connect([&]{
			if(current_thread) current_thread->join();
			if(debug_value.get_active()) {
				Threads::Mutex::Lock lock(mutex);
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
			progress_text.hide();
			run->set_sensitive(valid);
			stop->set_sensitive(false);
			current_thread = nullptr;
			if(!have_run) validate();
		});
	}

	void validate() {
		have_run = false;
		valid = true;
		valid &= !p->kernel_sizes.empty();
		valid &= input_image;
		run->set_sensitive(valid);
		if(valid && auto_run->get_active() && !current_thread)
			do_run();
	}

	// Run the algorithm in a new thread.
	void do_run() {
		run->set_sensitive(false);
		stop->set_sensitive(true);
		progress.set_fraction(0);
		progress.show();
		progress_text.set_text("Starting");
		progress_text.show();
		current_log.clear();

		// start thread
		continue_run = true;
		current_thread = Threads::Thread::create([=/* & doesn't work with threads */]{
			// The image to process
			auto input = pixbuf_to_multi_array(input_image);
			p->set_size(input.shape());
			auto run_p = p->runner();
			have_run = true;
			run_p->progress_cb = [=](double p, string d) {
				{
					Threads::Mutex::Lock lock(mutex);
					progress_value = p;
					progress_desc = d;
				}
				update_progress();
			};
			run_p->current_cb = [=](const boost::multi_array<T,2> &a, size_t s) {
				if(live_display_value.get_active()) {
					{
						Threads::Mutex::Lock lock(mutex);
						output_image = multi_array_to_pixbuf(a, auto_range_value.get_active());
					}
					update_output();
				}
				return continue_run;
			};
			if(debug_value.get_active())
				run_p->debug_cb = [=](const boost::multi_array<T,2> &a, string desc) {
					Threads::Mutex::Lock lock(mutex);
					current_log.push_back(debug_state{multi_array_to_pixbuf(a), desc});
				};
			auto result = run_p->run(input);
			q_value.set_value(run_p->q);
			stddev_value.set_value(run_p->input_stddev);
			output_image = multi_array_to_pixbuf(result, auto_range_value.get_active());
			update_output();
			algorithm_done();
		});
	}

	void open(RefPtr<Pixbuf> image) {
		input_image = image;
		output_image.reset();
		render();
		validate();
	}

	void open(string filename) {
		open(Pixbuf::create_from_file(filename));
	}

	RefPtr<Pixbuf> zoom_image(const RefPtr<Pixbuf> &img) const {
		if(zoom_level == 1) return img;
		return img->scale_simple(img->get_width() * zoom_level, img->get_height() * zoom_level, INTERP_NEAREST);
	}

	void render() {
		zoom_in->set_sensitive(zoom_level < 8);
		zoom_out->set_sensitive(zoom_level > 1);
		input_image_view.set(zoom_image(input_image));
		hadjustment->set_upper(input_image->get_width() * zoom_level);
		vadjustment->set_upper(input_image->get_height() * zoom_level);
		if(output_image) {
			output_image_view.set(zoom_image(output_image));
			save_image->set_sensitive(true);
			notebook.set_current_page(1);
		} else {
			save_image->set_sensitive(false);
			notebook.set_current_page(0);
			output_image_view.clear();
		}
	}
};


using namespace Gio;
using namespace boost::program_options;

struct app_t : Gtk::Application {
	bool debug = false, gpu_available = false;
	string input_file;
	shared_ptr<main_window> main;

	shared_ptr<params<T>> p = make_shared<params<T>>();
	shared_ptr<vex::Context> clctx;

	app_t()
	: Gtk::Application("smre.main", APPLICATION_HANDLES_COMMAND_LINE | APPLICATION_NON_UNIQUE) {}

	int on_command_line(const RefPtr<ApplicationCommandLine> &cmd) {
		clctx = make_shared<vex::Context>(vex::Filter::Count(1) && vex::Filter::Env);
		if(*clctx) {
			vex::StaticContext<>::set(*clctx);
			cerr << "OpenCL: " << *clctx << endl;
			gpu_available = true;
			p->use_gpu = true;
			p->use_fft = false;
		} else {
			cerr << "OpenCL unavailable." << endl;
			p->use_gpu = false;
			p->use_fft = true;
		}

		string output_file;
		bool dump_steps;

		options_description main_desc("Options");
		main_desc.add_options()
			("help,h", "show this help")
			("debug,d", bool_switch(&debug), "dump intermediate steps")
			("input,i", value(&input_file)->value_name("<file>"),
				"input image file (png,jpeg,…)")
			("output,o", value(&output_file)->value_name("<file>"),
				"output image file (png) → disables GUI")
			("dump-steps", bool_switch(&dump_steps),
				"save output after each step to “<output>.step<int>.png” (CLI only)");
		if(gpu_available) main_desc.add_options()
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
				"Resolvent function to use, either “L2” for L² or “H1 <delta>” for H¹")
			("max-steps,#", value(&p->max_steps)->default_value(p->max_steps)->value_name("<int>"),
				"Maximum number of optimization steps")
			("tolerance,e", value(&p->tolerance)->default_value(p->tolerance)->value_name("<float>"),
				"Stop when inverse relative change larger than this value")
			("tau,t", value(&p->tau)->default_value(p->tau)->value_name("<float>"),
				"Step size τ (large)")
			("sigma,s", value(&p->sigma)->default_value(p->sigma)->value_name("<float>"),
				"Step size σ (small)")
			("std", value(&p->input_stddev)->default_value(p->input_stddev)->value_name("<float>"),
				"Set the input image standard deviation explicitly instead of guessing it");

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

		options_description desc("Environment variables:\n"
			"  OMP_NUM_THREADS=<int>  Number of threads to use for CPU (default: 1/core)\n"
			"  OCL_DEVICE=<name>      OpenCL device to use as GPU");
		desc.add(main_desc).add(par_desc).add(q_desc);
		positional_options_description pos;
		pos.add("input", -1);
		int argc;
		char **argv = cmd->get_arguments(argc);
		variables_map vm;
		store(command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
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
			cerr << setprecision(5) << setw(8) << (p * 100) << "%  ";
			const size_t w = 20;
			const size_t b = p * w;
			for(size_t i = 0 ; i < w ; i++) cerr << (i < b ? "█" : "▁");
			cerr << "  " << d << "\r" << flush;
		};
		if(debug) {
			size_t i = 0;
			auto fmt = boost::format("%s.%05d.%s.png");
			run_p->debug_cb = [&](const boost::multi_array<T,2> &a, string desc) {
				multi_array_to_pixbuf(a)->save(str(fmt % output_file % (i++) % desc), "png");
			};
		}
		if(dump_steps) {
			auto fmt = boost::format("%s.step%05d.png");
			run_p->current_cb = [&](const boost::multi_array<T, 2> &a, size_t step) {
				multi_array_to_pixbuf(a)->save(str(fmt % output_file % step), "png");
				return true;
			};
		}
		auto output = run_p->run(input);
		multi_array_to_pixbuf(output)->save(output_file, "png");

		return EXIT_SUCCESS;
	}

	void on_activate() {
		main = make_shared<main_window>(p, debug, gpu_available);
		add_window(*main);
		if(!input_file.empty()) main->open(input_file);
		main->show();
	}
};

int main(int argc, char **argv) {
	try {
		return app_t().run(argc, argv);
	} catch(cl::Error &e) {
		cerr << "OpenCL error: " << e << endl;
	}
}
