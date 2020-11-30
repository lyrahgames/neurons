cxx.std = latest
using cxx

hxx{*}: extension = hpp
cxx{*}: extension = cpp

cxx.poptions =+ "-I$src_base" "-I/usr/include/eigen3"

exe{main}: cxx{main} hxx{**} $libs