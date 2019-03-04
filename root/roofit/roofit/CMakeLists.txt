RB_ADD_GBENCHMARK(RoofitBinnedBenchmark
  RooFitBinnedBenchmarks.cxx
  LABEL short
  LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory)

RB_ADD_GBENCHMARK(RoofitUnBinnedBenchmark
  RooFitUnBinnedBenchmarks.cxx
  LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory)

find_package(ZeroMQ REQUIRED)

RB_ADD_GBENCHMARK(RoofitMP RooFitMP.cxx
LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory RooFitZMQ)
target_include_directories(RoofitMP PRIVATE ${ZeroMQ_INCLUDE_DIR}) # FIXME: somehow, this does not get propagated from RooFitZMQ to RoofitMP

RB_ADD_GBENCHMARK(RoofitMPworkspace RooFitMP_workspace.cxx
        LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory RooFitZMQ)
target_include_directories(RoofitMPworkspace PRIVATE ${ZeroMQ_INCLUDE_DIR}) # FIXME: somehow, this does not get propagated from RooFitZMQ to RoofitMPworkspace

RB_ADD_GBENCHMARK(RoofitMPworkspaceNoOptConst RooFitMP_workspace_noOptConst.cxx
        LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory RooFitZMQ)
target_include_directories(RoofitMPworkspaceNoOptConst PRIVATE ${ZeroMQ_INCLUDE_DIR}) # FIXME: somehow, this does not get propagated from RooFitZMQ