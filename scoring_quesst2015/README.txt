This directory contains:

- Ground truth directory groundtruth_quesst2015_dev containing ecf, tlist
  and rttm files for the development dataset (needed by the scoring script):

      - quesst2015.ecf.xml
      - quesst2015_dev.rttm
      - quesst2015_dev.tlist.xml

- Ground truth directories groundtruth_quesst2015_dev_T1,
  groundtruth_quesst2015_dev_T2 and groundtruth_quesst2015_dev_T3,
  containing ecf, tlist and rttm files for the subsets of queries
  of types 1, 2 and 3, respectively. These directories allow us
  to measure system performance on a specific type of queries.

- Example STD directory, including an incomplete fake system
  for QUESST 2015 (called quesst2015.stdlist.xml):

      - example_QUESST2015_incomplete

- The TWV/Cnxe scoring script for Mediaeval QUESST 2015:

      - score-TWV-Cnxe.sh

- Scoring tool for Mediaeval QUESST 2015:

      - MediaEvalQUESST2015.jar

- Example STD directory, including a complete fake system
  (created from a real system submitted to Mediaeval SWS 2013):

      - example_STD

- A second example STD directory, including an incomplete fake system:

      - example_STD_incomplete 

  If an incomplete system is submitted, participants must provide
  a default score, which will be used to compute both the Cnxe value
  and TWV metrics.

- Example ground truth directory (created from real ground truth files
  used in Mediaeval SWS 2013):

      - example_groundtruth

- This README file


How to use the scoring script
=============================

1. Systems are expected to produce a complete set of detections
   (i.e. they produce a score for each trial) in a .stdlist.xml file
   with the format described in the NIST 2006 STD Evaluation Plan
   (see http://www.itl.nist.gov/iad/mig/tests/std/2006/).
   The fake system included in the example_STD directory can be used
   as template.

2. A directory must be created containing the .stdlist.xml file.

3. The directory where the scoring script is located must also contain
   the MediaEvalQUESST2015.jar tool.

4. The scoring command should be as follows:

   ./score-TWV-Cnxe.sh <STDLIST_dir> <Ground_Truth_dir> [<default_score>]

   For instance, to score the example system output stored in the
   example_STD directory, the following command must be run:

   ./score-TWV-Cnxe.sh example_STD example_groundtruth 

   To score the example incomplete system output stored in the
   example_STD_incomplete directory (using a default -10 score),
   the following command must be run:

   ./score-TWV-Cnxe.sh example_STD_incomplete example_groundtruth -10

   Finally, to score the incomplete system provided as example
   for QUESST 2015, using -10 as default score, we would run:

   ./score-TWV-Cnxe.sh example_QUESST2015_incomplete groundtruth_quesst2015_dev -10

   To score the same system on a subset of queries TX (T1, T2 or T3),
   first we must copy the quesst2015.stdlist.xml file to a new folder:

   mkdir example_QUESST2015_incomplete_TX

   cp example_QUESST2015_incomplete/quesst2015.stdlist.xml example_QUESST2015_incomplete_TX

   and then run the scoring script:

   ./score-TWV-Cnxe.sh example_QUESST2015_incomplete_TX groundtruth_quesst2015_dev_TX -10

5. Once the scoring script finishes, results (score.out, DET.pdf and TWV.pdf)
   are stored in the same directory where the .stdlist.xml file was located.

--------------------------------------------------------------------
   EXAMPLE OUTPUT OF THE SCORING TOOL (MediaEvalQUESST2015.jar)
   WHEN APPLIED TO THE COMPLETE EXAMPLE SYSTEM

   Please, check that you get the same output on your machine.

   The model-dependent calibration results correspond to using
   a per-query calibration, thus yielding an Upper-Bound TWV (UB-TWV)
   and a Lower-Bound Cnxe (LB-Cnxe).
--------------------------------------------------------------------

user@machine$ java -Xmx2g -jar MediaEvalQUESST2015.jar example_STD/MySystem.stdlist.xml example_groundtruth/GroundTruth.rttm example_groundtruth/Audio.ecf.xml example_groundtruth/Query.tlist.xml 0.0741289
Loading tlist file: example_groundtruth/Query.tlist.xml ...  Done (size: 497)
Loading ecf file: example_groundtruth/Audio.ecf.xml ...  Done (size: 10762)
Loading rttm file: example_groundtruth/GroundTruth.rttm ...  Done (size: 5079)
Removing unseen models metadata ...  Done (removed: 2)
Loading stdlist file: example_STD/MySystem.stdlist.xml ... Done
Scores: 5327190 (495 model x 10762 test)  given: 5327190  default(null): 0  minScore: -3.0  maxScore: 48.5092

Main Results:

actTWV: 0.40079603  maxTWV: 0.41911224  Threshold: 5.2125
actCnxe: 0.97036207  minCnxe: 0.6852714

Model-dependent calibration Results:

Upper-bound TWV (model-dependent offset)  -  actTWV: 0.5082787  maxTWV: 0.5082787  Threshold: 0.0
Lower-bound Cnxe (model-dependent PAV)    -  actCnxe: 0.46791807  minCnxe: 0.4677087
user@machine$
