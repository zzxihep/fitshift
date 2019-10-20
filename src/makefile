FLAG = -fPIC -std=c++11 -O2
PY_CFLAGS  := $(shell python-config --includes)
CC = g++
SHARE = -shared

objects = convol.o convol_wrap.o

default : _convol.so _rebin.so

_convol.so : $(objects)
	$(CC) -o _convol.so $(objects) $(SHARE)

convol.o : convol.cpp convol.h
	$(CC) -c convol.cpp $(FLAG)

convol_wrap.o : convol_wrap.cxx
	$(CC) -c convol_wrap.cxx $(FLAG) $(PY_CFLAGS)

convol_wrap.cxx : convol.i
	swig -python -c++ convol.i

_rebin.so : rebin.o rebin_wrap.o
	$(CC) -o _rebin.so rebin.o rebin_wrap.o $(SHARE)

rebin.o : rebin.cpp
	$(CC) -c rebin.cpp $(FLAG)

rebin_wrap.o : rebin_wrap.cxx
	$(CC) -c rebin_wrap.cxx $(FLAG) $(PY_CFLAGS)

rebin_wrap.cxx : rebin.i
	swig -python -c++ rebin.i

clean :
	rm _convol.so $(objects) convol_wrap.cxx convol.py \
		_rebin.so rebin.o rebin_wrap.o rebin_wrap.cxx rebin.py
