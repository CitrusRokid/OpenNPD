DAR_ROOTDIR=$(shell pwd)/../
DAR_BINDIR=$(shell pwd)/../bin/
DAR_OBJDIR=$(shell pwd)/../.obj/

export DAR_ROOTDIR DAR_BINDIR DAR_OBJDIR

SUBDIRS=$(shell ls -l | grep ^d | awk '{print $$(NF)}')

.PHONY: all 

all: $(SUBDIRS)

$(SUBDIRS):ECHO
	$(MAKE) -C $@

ECHO:
	@echo $(SUBDIRS)


clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) clean -C $$dir; \
		done
