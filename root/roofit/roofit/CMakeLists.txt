RB_ADD_GBENCHMARK(RoofitBinnedBenchmark
  RooFitBinnedBenchmarks.cxx
  LABEL short
  LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory)

RB_ADD_GBENCHMARK(RoofitUnBinnedBenchmark
  RooFitUnBinnedBenchmarks.cxx
  LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory)

find_package(ZeroMQ REQUIRED)

RB_ADD_GBENCHMARK(RoofitMultiproc
  RooFitMultiproc.cxx
  LIBRARIES Core Hist MathCore RIO RooFit RooStats RooFitCore HistFactory RooFitZMQ)
target_include_directories(RoofitMultiproc PRIVATE ${ZeroMQ_INCLUDE_DIR}) # FIXME: somehow, this does not get propagated from RooFitZMQ to RoofitMultiproc