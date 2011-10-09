
all : build
	cd build ; make && make test

build :
	mkdir build ; cd build ; cmake ..

clean :
	-rm -rf build

new : clean all
