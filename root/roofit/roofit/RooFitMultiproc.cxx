#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooAddModel.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "TH1.h"
#include "TRandom.h"
#include "RooGaussian.h"
#include "RooGaussModel.h"
#include "RooCategory.h"
#include "RooBMixDecay.h"
#include "RooBCPEffDecay.h"
#include "RooBDecay.h"
#include "RooDecay.h"
#include "RooFormulaVar.h"
#include "RooTruthModel.h"
#include "TCanvas.h"
#include "RooPlot.h"
#include "RooMinimizer.h"
#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/HistoToWorkspaceFactoryFast.h"
#include "RooStats/ModelConfig.h"
#include "RooLinkedListIter.h"
#include "RooRealSumPdf.h"
#include "RooNLLVar.h"
#include "RooAddition.h"
//#include <MultiProcess/NLLVar.h>


using namespace RooFit ;

#include "benchmark/benchmark.h"

//using namespace RooFit;
//using namespace RooStats;
//using namespace HistFactory;

using namespace RooStats;
using namespace HistFactory;

Sample addVariations(Sample asample, int nnps, bool channel_crosstalk, int channel)
{
   for (int nuis = 0; nuis < nnps; ++nuis) {
      TRandom *R = new TRandom(channel * nuis / nnps);
      Double_t random = R->Rndm();
      double uncertainty_up = (1 + random) / sqrt(100);
      double uncertainty_down = (1 - random) / sqrt(100);
      std::cout << "in channel " << channel << "nuisance +/- [" << uncertainty_up << "," << uncertainty_down << "]"
                << std::endl;
      std::string nuis_name = "norm_uncertainty_" + std::to_string(nuis);
      if (!channel_crosstalk) {
         nuis_name = nuis_name + "_channel_" + std::to_string(channel);
      }
      asample.AddOverallSys(nuis_name, uncertainty_up, uncertainty_down);
   }
   return asample;
}

Channel makeChannel(int channel, int nbins, int nnps, bool channel_crosstalk)
{
   std::string channel_name = "Region" + std::to_string(channel);
   Channel chan(channel_name);
   auto Signal_Hist = new TH1F("Signal", "Signal", nbins, 0, nbins);
   auto Background_Hist = new TH1F("Background", "Background", nbins, 0, nbins);
   auto Data_Hist = new TH1F("Data", "Data", nbins, 0, nbins);
   for (Int_t bin = 1; bin <= nbins; ++bin) {
      for (Int_t i = 0; i <= bin; ++i) {
         Signal_Hist->Fill(bin + 0.5);
         Data_Hist->Fill(bin + 0.5);
      }
      for (Int_t i = 0; i <= nbins; ++i) {
         Background_Hist->Fill(bin + 0.5);
         Data_Hist->Fill(bin + 0.5);
      }
   }
   chan.SetData(Data_Hist);
   Sample background("background");
   background.SetNormalizeByTheory(false);
   background.SetHisto(Background_Hist);
   background.ActivateStatError();
   Sample signal("signal");
   signal.SetNormalizeByTheory(false);
   signal.SetHisto(Signal_Hist);
   signal.ActivateStatError();
   signal.AddNormFactor("SignalStrength", 1, 0, 3);

   if (nnps > 0) {
      signal = addVariations(signal, nnps, true, channel);
      background = addVariations(background, nnps, false, channel);
   }
   chan.AddSample(background);
   chan.AddSample(signal);
   return chan;
}

void buildBinnedTest(int n_channels = 1, int nbins = 10, int nnps = 1, const char *name_rootfile = "")
{
  std::cout<<"in build binned test with output"<<name_rootfile<<std::endl;
   bool channel_crosstalk = true;
   Measurement meas("meas", "meas");
   meas.SetPOI("SignalStrength");
   meas.SetLumi(1.0);
   meas.SetLumiRelErr(0.10);
   meas.AddConstantParam("Lumi");
   Channel chan;
   for (int channel = 0; channel < n_channels; ++channel) {
     chan = makeChannel(channel, nbins, nnps, channel_crosstalk);
      meas.AddChannel(chan);
   }
   HistoToWorkspaceFactoryFast hist2workspace(meas);
   RooWorkspace *ws;
   if (n_channels < 2) {
      ws = hist2workspace.MakeSingleChannelModel(meas, chan);
   } else {
      ws = hist2workspace.MakeCombinedModel(meas);
   }
   RooFIter iter = ws->components().fwdIterator();
   RooAbsArg *arg;
   while ((arg = iter.next())) {
      if (arg->IsA() == RooRealSumPdf::Class()) {
         arg->setAttribute("BinnedLikelihood");
         std::cout << "component " << arg->GetName() << " is a binned likelihood" << std::endl;
      }
   }
   ws->SetName("BinnedWorkspace");
   ws->writeToFile(name_rootfile);
}

//############## End of Base Algorithms ##############################
//####################################################################
//############## Start Of #Channel tests #############################

static void BM_RooFit_BinnedMPFE(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Minimization);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Eval);
   int bins = state.range(0);
   int cpu = state.range(1);
   TFile *infile = new TFile("workspace.root","RECREATE");
   //   if (infile->IsZombie()) {
   buildBinnedTest(1, bins, 0, "workspace.root");
   std::cout << "Workspace for tests was created!" << std::endl;
   //}
   infile = TFile::Open("workspace.root");
   RooWorkspace *w = static_cast<RooWorkspace *>(infile->Get("BinnedWorkspace"));
   RooAbsData *data = w->data("obsData");
   ModelConfig *mc = static_cast<ModelConfig *>(w->genobj("ModelConfig"));
   RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());
   RooAbsReal *nll = pdf->createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setProfile(0);
   m.setLogFile("benchmigradnchannellog");
   for (auto _ : state) {
      m.migrad();
   }
   delete data;
   delete infile;
   delete mc;
   delete pdf;
   delete nll;
}


static void BM_RooFit_BinnedMultiProc(benchmark::State &state)
{
   std::cout << "About to run Binned MultiProcess"<< std::endl;
   gErrorIgnoreLevel = kInfo;
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Minimization);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Eval);
   int bins = state.range(0);
   int cpu = state.range(1);
   TFile *infile = new TFile("workspace.root","RECREATE");
   //   if (infile->IsZombie()) {
   buildBinnedTest(1, bins, 0, "workspace.root");
   std::cout << "Workspace for tests was created!" << std::endl;
   //}
   infile = TFile::Open("workspace.root");
   RooWorkspace *w = static_cast<RooWorkspace *>(infile->Get("BinnedWorkspace"));
   RooAbsData *data = w->data("obsData");
   ModelConfig *mc = static_cast<ModelConfig *>(w->genobj("ModelConfig"));
   RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());
   RooAbsReal *nll = pdf->createNLL(*data);

   std::size_t NumCPU = cpu;
   if(dynamic_cast<RooAddition*>(nll)) { std::cout << "indeed, nll is a RooAddition" << std::endl;}

   std::cout << "About to initialise MultiProcess"<< std::endl;
   MultiProcess::NLLVar nll_mp(NumCPU, MultiProcess::NLLVarTask::bulk_partition, *dynamic_cast<RooNLLVar*>(nll));
   std::cout << "Success! Now on to the minimisation..."<< std::endl;

   RooMinimizer m(nll_mp);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setProfile(0);
   m.setLogFile("benchmigradnchannellog");
   std::cout << "About to run Migrad"<< std::endl;   
   for (auto _ : state) {
      m.migrad();
   }
   delete data;
   delete infile;
   delete mc;
   delete pdf;
   delete nll;
}

static void BM_RooFit_BDecayMultiproc(benchmark::State &state)
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

   std::size_t NumCPU = cpu;

   // Create NLL
   RooAbsReal *nll = bmix.createNLL(*data);//, NumCPU(cpu, 0));
   MultiProcess::NLLVar nll_mp(NumCPU, MultiProcess::NLLVarTask::bulk_partition, *dynamic_cast<RooNLLVar*>(nll));

   RooMinimizer m(nll_mp);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setLogFile("benchmigradnchanellog");
   while (state.KeepRunning()) {
      m.migrad();
   }
   delete data;
   delete nll;
}
static void BM_RooFit_BDecayMPFE(benchmark::State &state)
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

static void BM_RooFit_BDecayGaussResolution(benchmark::State &state)
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
   // Build a gaussian resolution model
   RooRealVar bias1("bias1","bias1",0, -1., 1.) ;
   RooRealVar sigma1("sigma1","sigma1", 1, 0.1, 1.9) ;
   RooGaussModel gm1("gm1","gauss model 1",dt,bias1,sigma1) ;

   // Construct decay(t) (x) gauss1(t)
   RooDecay decay_gm1("decay_gm1","decay",dt,tau,gm1,RooDecay::DoubleSided) ;

   // Generate Some Data
   RooDataSet* data = decay_gm1.generate(RooArgSet(dt,mixState,tagFlav),events);

   // Create NLL
   RooAbsReal *nll = decay_gm1.createNLL(*data, NumCPU(cpu, 0));
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

static void BM_RooFit_BDecayDoubleGauss(benchmark::State &state)
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
   // Build a gaussian resolution model
   RooRealVar bias1("bias1","bias1",0, -1., 1.) ;
   RooRealVar sigma1("sigma1","sigma1", 1, 0.1, 1.9) ;
   RooGaussModel gm1("gm1","gauss model 1",dt,bias1,sigma1) ;

   // Construct decay(t) (x) gauss1(t)
   RooDecay decay_gm1("decay_gm1","decay",dt,tau,gm1,RooDecay::DoubleSided) ;

   // Build another gaussian resolution model
   RooRealVar bias2("bias2","bias2",0) ;
   RooRealVar sigma2("sigma2","sigma2",5) ;
   RooGaussModel gm2("gm2","gauss model 2",dt,bias2,sigma2) ;
   // Build a composite resolution model f*gm1+(1-f)*gm2
   RooRealVar gm1frac("gm1frac","fraction of gm1",0.5) ;
   RooAddModel gmsum("gmsum","sum of gm1 and gm2",RooArgList(gm1,gm2),gm1frac) ;
   // Construct decay(t) (x) (f*gm1 + (1-f)*gm2)
   RooDecay decay_gmsum("decay_gmsum","decay",dt,tau,gmsum,RooDecay::DoubleSided) ;

   // Generate Some Data
   RooDataSet* data = decay_gmsum.generate(RooArgSet(dt,mixState,tagFlav),events);

   // Create NLL
   RooAbsReal *nll = decay_gmsum.createNLL(*data, NumCPU(cpu, 0));
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


static void EventArguments(benchmark::internal::Benchmark* b) {
  for (int i = 1; i <=10; ++i )
    for (int j = 1; j <= 3; ++j)
      b->Args({i*1000, j});
}

BENCHMARK(BM_RooFit_BDecayMultiproc)->Apply(EventArguments)->UseRealTime()->Iterations(12);
BENCHMARK(BM_RooFit_BDecayMPFE)->Apply(EventArguments)->UseRealTime()->Iterations(12);
//BENCHMARK(BM_RooFit_BDecayGaussResolution)->Apply(EventArguments)->UseRealTime()->Iterations(12);
//BENCHMARK(BM_RooFit_BDecayDoubleGauss)->Apply(EventArguments)->UseRealTime()->Iterations(12);

static void BinArguments(benchmark::internal::Benchmark* b) {
  for (int i = 1; i <= 10; ++i)
    for (int j = 1; j <= 3; ++j)
      b->Args({i+5, j});
}

BENCHMARK(BM_RooFit_BinnedMPFE)->Apply(BinArguments)->UseRealTime()->Unit(benchmark::kMicrosecond)->Iterations(10);
//BENCHMARK(BM_RooFit_BinnedMultiProc)->Apply(BinArguments)->UseRealTime()->Unit(benchmark::kMicrosecond)->Iterations(10);

BENCHMARK_MAIN();
