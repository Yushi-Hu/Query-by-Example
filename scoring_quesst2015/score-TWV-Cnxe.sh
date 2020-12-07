#!/bin/bash -e 

# Expects 3 arguments:
#
# $1: directory with stdlist-file (output files will be put in here)
# $2: directory with ground-truth (rttm, ecf and tlist) files
# $3: (optional) default score for non reported trials (incomplete system)

STD=$1
GT=$2
DS=$3

# MediaEvalQUESST2015.jar (we assume that it is located in the working directory)

JAR=MediaEvalQUESST2015.jar

P_target=0.0008
C_fa=1
C_miss=100

EFFPRIOR=`echo "scale=7; ($C_miss * $P_target)/($C_miss * $P_target + $C_fa * (1 - $P_target))" | bc`

java -Xmx2g -jar $JAR $STD/*.stdlist.xml $GT/*.rttm $GT/*.ecf.xml $GT/*.tlist.xml $EFFPRIOR $DS | tee $STD/score.out

test ${PIPESTATUS[0]} -eq 0

# The following code creates the graphs (.pdf files)
# It assumes that the gnuplot and ps2pdf commands are available
gnuplot plot.DET.plt | ps2pdf - $STD/DET.pdf
gnuplot plot.TWV.plt | ps2pdf - $STD/TWV.pdf
rm plot.*
