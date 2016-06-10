
LINK = libtool --mode=link --tag=CXX g++

C := gcc
CXX := g++

AR      := ar
ARFLAGS := crv
RANLIB  := ranlib
SHARED  := -shared
SRC_DIR := 
LIBCXXFLAGS := -fno-strict-aliasing -fPIC  -fopenmp -fpermissive -Werror
LIBCFLAGS := -fPIC -fopenmp -Werror

CFLAGS = -Wall ${LIBCFLAGS}
CXXFLAGS = -Wall ${LIBCXXFLAGS}

DLIB = 0

# Compiler switches. Default -g.
ifeq ($(RELEASE), 1)
    CFLAGS += -O3 -DNDEBUG -DSHMEM -DDLIB_PNG_SUPPORT -DDLIB_JPEG_SUPPORT
    CXXFLAGS += -O3 -DNDEBUG -DSHMEM -DDLIB_PNG_SUPPORT -DDLIB_JPEG_SUPPORT
    DLIBSOURCE = 
else
    CFLAGS +=  -DDEBUG  -pg -g3 -DSHMEM -DDLIB_PNG_SUPPORT -DDLIB_JPEG_SUPPORT
    CXXFLAGS +=  -DDEBUG  -pg -g3 -DSHMEM -DDLIB_PNG_SUPPORT -DDLIB_JPEG_SUPPORT
    ifeq ($(DLIB), 1)
        DLIBSOURCE = ../../3rdparty/dlib/dlib/all/source.cpp
    endif
endif

# Compiler switches. Defult no multi thread support
ifeq ($(MTHREADS), 1)
    CFLAGS += -DMULTI_THREAD_SUPPORT
    CXXFLAGS += -DMULTI_THREAD_SUPPORT
    EXTRALIBS += -pthread
endif

INC := \
	-I. \
	-I../../include/ \
	-I../../3rdparty/opencv/include \
	-I../../3rdparty/opencv/include/opencv \
	-I/usr/local/include/ \
	-I../npd_train/

CFLAGS += $(INC)
CXXFLAGS += $(INC)

LDDIR := \
	-L../../lib/ \
	-L../../3rdparty/opencv/lib/ \
	-L/usr/local/lib/

OPENCVLIBS= -lopencv_highgui -lopencv_core \
			-lopencv_imgproc -lopencv_gpu \
			-lopencv_contrib -lopencv_flann \
			-lopencv_calib3d -lopencv_ml \
			-lopencv_video -lopencv_legacy \
			-lopencv_objdetect -lopencv_photo \
			-lopencv_ocl -lopencv_nonfree \
			-lopencv_features2d

ifeq ($(DLIB), 1)
    DLIBLIBS = -ldlib 
else
    DLIBLIBS = 
endif

EXTRALIBS= $(OPENCVLIBS) -lnpd

OBJS = $(patsubst %.cpp,%.o,$(wildcard *.cpp))

.PHONY: all 

all:  $(OBJS) test_cam test_detect

test_detect: test_detect.o $(DLIBSOURCE)
	$(CXX) $(CXXFLAGS) $(LDDIR) -o ../../bin/$@ $^ $(EXTRALIBS)

test_cam: test_cam.o $(DLIBSOURCE)
	$(CXX) $(CXXFLAGS) $(LDDIR) -o ../../bin/$@ $^ $(EXTRALIBS)  -fopenmp -lpthread

clean:
	rm -rf *.o
