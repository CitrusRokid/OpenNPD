%% This script shows how to compile the mex functions in this package, 
% especially how to enable the -openmp option, which is important for speedup.

fprintf('Start to compile mex functions.\n');

winStr = '-largeArrayDims -DWINDOWS ''COMPFLAGS=/openmp $COMPFLAGS'' -outdir ../bin/';
lnxStr = '-largeArrayDims -DLINUX ''CFLAGS=\$CFLAGS -fopenmp'' ''LDFLAGS=\$LDFLAGS -fopenmp'' -outdir ../bin/';
comStr = '-largeArrayDims -outdir ../bin/';

os = computer;

if strcmp(os, 'PCWIN') || strcmp(os, 'PCWIN64')
    eval(['mex ', winStr, ' LearnDQT.cpp']);
    eval(['mex ', winStr, ' NPD.cpp']);
    eval(['mex ', winStr, ' NPDClassify.cpp']);
    eval(['mex ', winStr, ' NPDGrid.cpp']);
    eval(['mex ', winStr, ' NPDScan.cpp']);
else if strcmp(os, 'GLNXA') || strcmp(os, 'GLNXA64')
        eval(['mex ', lnxStr, ' LearnDQT.cpp']);
        eval(['mex ', lnxStr, ' NPD.cpp']);
        eval(['mex ', lnxStr, ' NPDClassify.cpp']);
        eval(['mex ', lnxStr, ' NPDGrid.cpp']);
        eval(['mex ', lnxStr, ' NPDScan.cpp']);
    else
        % if you know how to enable the openmp, please config it by
        % yourself, otherwise the speed will be slow.
        eval(['mex ', comStr, ' LearnDQT.cpp']);
        eval(['mex ', comStr, ' NPD.cpp']);
        eval(['mex ', comStr, ' NPDClassify.cpp']);
        eval(['mex ', comStr, ' NPDGrid.cpp']);
        eval(['mex ', comStr, ' NPDScan.cpp']);
    end
end

fprintf('Compiling finished.\n\n');
