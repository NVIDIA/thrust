CC         = nvcc
CFLAGS     = -c -O2
LDFLAGS    =
SOURCES    = $(wildcard *.cu) 
SOURCES   += $(wildcard cuda/*.cu) 
OBJECTS    = $(SOURCES:.cu=.o)
INCLUDES   = -I../
EXECUTABLE = tester

all: $(SOURCES) $(EXECUTABLE)

clean: 
	$(RM) $(OBJECTS) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

%.o: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) $< -o $@

