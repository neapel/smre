#include <gtkmm.h>
#include <iostream>
#include "config.h"

using namespace std;
using namespace Gtk;
using namespace Glib;

static void on_open_image_activate() {
	cout << "open image!" << endl;
}

static void on_run_activate() {
	cout << "run" << endl;
}


int main(int argc, char **argv) {
	auto app = Application::create(argc, argv, "smre.main");
	auto builder = Builder::create();
	builder->add_from_file("gui.glade");

	Window *main_window; builder->get_widget("mainWindow", main_window);

	// Actions
	auto open_image = RefPtr<Action>::cast_dynamic(builder->get_object("openImage"));
	open_image->signal_activate().connect(sigc::ptr_fun(on_open_image_activate));
	auto run = RefPtr<Action>::cast_dynamic(builder->get_object("run"));
	run->signal_activate().connect(sigc::ptr_fun(on_run_activate));

	// Widgets
	SpinButton *tau; builder->get_widget("tau", tau);

	ComboBoxText *kernel; builder->get_widget("kernel", kernel);
	SpinButton *size; builder->get_widget("size", tau);
	SpinButton *min; builder->get_widget("min", tau);
	SpinButton *max; builder->get_widget("max", tau);

	// Kernel types
	kernel->append("gaussian");
	kernel->append("box");

	app->run(*main_window);


}
