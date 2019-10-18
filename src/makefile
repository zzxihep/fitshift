FLAG = -fPIC -std=c++11 -O2
HP = -I /usr/include/python2.7
SHARE = -shared

objects = convol.o convol_wrap.o

default : _convol.so

_convol.so : $(objects)
	g++ -o _convol.so $(objects) $(SHARE)

convol.o : convol.cpp convol.h
	g++ -c convol.cpp $(FLAG)

convol_wrap.o : convol_wrap.cxx
	g++ -c convol_wrap.cxx $(FLAG) $(HP)

convol_wrap.cxx : convol.i
	swig -python -c++ convol.i

clean :
	rm _convol.so $(objects) convol_wrap.cxx convol.py
