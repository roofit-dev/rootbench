#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooCategory.h"
#include "RooBMixDecay.h"
#include "RooBCPEffDecay.h"
#include "RooBDecay.h"
#include "RooFormulaVar.h"
#include "RooTruthModel.h"
#include "TCanvas.h"
#include "RooPlot.h"
#include "RooMinimizer.h"

using namespace RooFit ;

#include "benchmark/benchmark.h"

//using namespace RooFit;
//using namespace RooStats;
//using namespace HistFactory;

static void BM_RooFit_BDecayWithMixing(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   int events = state.range(0);
   int cpu = state.range(1);

   RooRealVar dt("dt","dt",-10,10) ;
   dt.setBins(40) ;
   // Parameters
   RooRealVar dm("dm","delta m(B0)",0.472,0.1,0.9) ;
   RooRealVar tau("tau","tau (B0)",1.547,1.3,1.9) ;
   RooRealVar w("w","flavour mistag rate",0.1) ;
   RooRealVar dw("dw","delta mistag rate for B0/B0bar",0.1) ;

   RooCategory mixState("mixState","B0/B0bar mixing state") ;
   mixState.defineType("mixed",-1) ;
   mixState.defineType("unmixed",1) ;

   RooCategory tagFlav("tagFlav","Flavour of the tagged B0") ;
   tagFlav.defineType("B0",1) ;
   tagFlav.defineType("B0bar",-1) ;

   // Use delta function resolution model
   RooTruthModel tm("tm","truth model",dt) ;
   // Construct Bdecay with mixing
   RooBMixDecay bmix("bmix","decay",dt,mixState,tagFlav,tau,dm,w,dw,tm,RooBMixDecay::DoubleSided) ;
   // Generate Some Data
   RooDataSet* data = bmix.generate(RooArgSet(dt,mixState,tagFlav),events);

   // Create NLL
   RooAbsReal *nll = bmix.createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setLogFile("benchmigradnchanellog");
   while (state.KeepRunning()) {
      m.migrad();
   }
   delete data;
   delete nll;
}


static void ChanArguments(benchmark::internal::Benchmark* b) {
  for (int i = 1; i <11<<10; i+=1000)
    for (int j = 1; j <= 4; ++j)
      b->Args({i, j});
}

BENCHMARK(BM_RooFit_BDecayWithMixing)->Apply(ChanArguments)->UseRealTime()->Iterations(12);

BENCHMARK_MAIN();
