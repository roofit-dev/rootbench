RB_ADD_GBENCHMARK(RoofitBinnedBenchmark
  RooFitBinnedBenchmarks.cxx
  LABEL long
  LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory)

RB_ADD_GBENCHMARK(RoofitUnBinnedBenchmark
  RooFitUnBinnedBenchmarks.cxx
  LABEL long
  LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory)

find_package(ZeroMQ REQUIRED)

RB_ADD_GBENCHMARK(RoofitMP RooFitMP.cxx
        LABEL long
        LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory RooFitZMQ)
target_include_directories(RoofitMP PRIVATE ${ZeroMQ_INCLUDE_DIR}) # FIXME: somehow, this does not get propagated from RooFitZMQ to RoofitMP

RB_ADD_GBENCHMARK(RoofitMPworkspace RooFitMP_workspace.cxx
        LABEL long
        LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory RooFitZMQ)
target_include_directories(RoofitMPworkspace PRIVATE ${ZeroMQ_INCLUDE_DIR}) # FIXME: somehow, this does not get propagated from RooFitZMQ to RoofitMPworkspace

RB_ADD_GBENCHMARK(RoofitMPworkspaceNoOptConst RooFitMP_workspace_noOptConst.cxx
        LABEL long
        LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory RooFitZMQ)
target_include_directories(RoofitMPworkspaceNoOptConst PRIVATE ${ZeroMQ_INCLUDE_DIR}) # FIXME: somehow, this does not get propagated from RooFitZMQ

RB_ADD_GBENCHMARK(RoofitMPworkspaceNumCPUInConfigFile RooFitMP_workspace_NumCPUInConfigFile.cxx
        LABEL long
        LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory RooFitZMQ)
target_include_directories(RoofitMPworkspaceNumCPUInConfigFile PRIVATE ${ZeroMQ_INCLUDE_DIR}) # FIXME: somehow, this does not get propagated from RooFitZMQ