GCC ?= g++

CCFLAGS := -std=c++11

LDFLAGS += $(shell pkg-config --libs opencv)
INCLUDES += $(shell pkg-config --cflags opencv)

SRCS=VideoCapture.cpp
OBJS=VideoCapture.o
RM=rm -f
EXEC=app

# Target rules
all: build

build: ${EXEC}

%.o: %.cpp %.h
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

${EXEC}: ${OBJS}
	$(GCC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	${RM} ${EXEC} *.o 

dist-clean:
	$(RM) $(EXEC)
