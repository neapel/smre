
all : releaseb debugb

releaseb : release
	cd release ; make -j8

debugb : debug
	cd debug ; make -j8

release :
	mkdir -p release ; cd release ; cmake -DCMAKE_BUILD_TYPE=Release ..

debug :
	mkdir -p debug ; cd debug ; cmake -DCMAKE_BUILD_TYPE=Debug ..

clean :
	-rm -rf release debug

new : clean all
