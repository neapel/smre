
all : build
	cd build ; make -j8

build :
	mkdir -p build ; cd build ; cmake -DCMAKE_BUILD_TYPE=Debug ..

clean :
	-rm -rf build

new : clean all
