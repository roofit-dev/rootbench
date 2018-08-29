#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooAddModel.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "TH1.h"
#include "TRandom.h"
#include "RooArgList.h"
#include "RooAddPdf.h"
#include "RooCategory.h"
#include "RooPolynomial.h"
#include "RooFormulaVar.h"
#include "RooParamHistFunc.h"
#include "RooHistConstraint.h"
#include "RooRealSumPdf.h"
#include "RooProdPdf.h"
#include "RooDataHist.h"
#include "TCanvas.h"
#include "RooPlot.h"
#include "RooMinimizer.h"
#include "RooStats/RooStatsUtils.h"
#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/HistoToWorkspaceFactoryFast.h"
#include "RooStats/ModelConfig.h"
#include "RooLinkedListIter.h"
#include "RooRealSumPdf.h"
#include "RooNLLVar.h"
#include "RooAddition.h"
// #include <MultiProcess/NLLVar.h>

#include "benchmark/benchmark.h"
using namespace RooFit ;
using namespace RooStats;
using namespace HistFactory;

void buildBinnedTest(int nbins = 10, int nnps=0, const char *name_rootfile = "")
{
   bool channel_crosstalk = true;
   Measurement meas("meas", "meas");
   meas.SetPOI("SignalStrength");
   meas.SetLumi(1.0);
   meas.SetLumiRelErr(0.10);
   meas.AddConstantParam("Lumi");
   std::string channel_name = "RegionOne";
   Channel chan(channel_name);
   std::cout << "Made channel let's go buffalo! Nbins:" <<nbins<< ", NP mode:"<< nnps<<std::endl;
   RooRealVar x("x","x",0,nbins) ;
   x.setBins(nbins) ;
   // Parameters
   RooRealVar a0("a0","a0",0) ;
   RooRealVar a1("a1","a1",1,0,2) ;
   RooRealVar a2("a2","a2",0);

   RooPolynomial p0("p0","p0",x);
   RooPolynomial p1("p1","p1",x,RooArgList(a0,a1,a2),0);

   RooDataHist *dh_bkg = p0.generateBinned(x, 100000);
   RooDataHist *dh_sig = p1.generateBinned(x, 10000);
   RooRealVar bkgfrac("bkgfrac","fraction of background",0.1,0.,1.);
   RooAddPdf model("model","sig+bkg",RooArgList(p1, p0),bkgfrac);
   RooDataHist *dh_data = model.generateBinned(x, 110000);

   Sample background("background");
   background.SetNormalizeByTheory(false);
   background.SetHisto( dh_bkg->createHistogram("h_bkg", x));
   Sample signal("signal");
   signal.SetNormalizeByTheory(false);
   signal.SetHisto( dh_sig->createHistogram("h_sig", x));
   signal.AddNormFactor("SignalStrength", 1, 0, 5);

   if (nnps == 1){
     a1.setVal(2);
     RooDataHist *dh_sig_up = p1.generateBinned(x, 1100000000);
     dh_sig_up->SetName("dh_sig_up");
     a1.setVal(.5);
     RooDataHist *dh_sig_down = p1.generateBinned(x, 900000000);
     dh_sig_down->SetName("dh_sig_down");
     HistoSys signal_shape("SignalShape");
     signal_shape.SetHistoHigh(dh_sig_up->createHistogram("h_sig_up",x));
     signal_shape.SetHistoLow(dh_sig_down->createHistogram("h_sig_down",x));
     signal.AddHistoSys( signal_shape );
   } else if(nnps == 2){
     signal.AddOverallSys( "SignalNorm",  0.9, 1.1 );
   } else if(nnps == 3){
     signal.ActivateStatError();
   } else if(nnps == 4){
     signal.ActivateStatError();
     a1.setVal(2);
     RooDataHist *dh_sig_up = p1.generateBinned(x, 1100000000);
     dh_sig_up->SetName("dh_sig_up");
     a1.setVal(.5);
     RooDataHist *dh_sig_down = p1.generateBinned(x, 900000000);
     dh_sig_down->SetName("dh_sig_down");
     HistoSys signal_shape("SignalShape");
     signal_shape.SetHistoHigh(dh_sig_up->createHistogram("h_sig_up",x));
     signal_shape.SetHistoLow(dh_sig_down->createHistogram("h_sig_down",x));
     signal.AddHistoSys( signal_shape );
   } else if(nnps == 5 or nnps == 6){
     signal.ActivateStatError();
   } 

   chan.AddSample(background);
   chan.AddSample(signal);

   chan.SetData( dh_data->createHistogram("h_data", x) );

   meas.AddChannel(chan);
   
   RooWorkspace *ws;

   if (nnps == 5){
     chan.SetName("RegionTwo");
     meas.AddChannel(chan);
     HistoToWorkspaceFactoryFast hist2workspace(meas);
     ws = hist2workspace.MakeCombinedModel(meas);
   } else {
     HistoToWorkspaceFactoryFast hist2workspace(meas);
     ws = hist2workspace.MakeSingleChannelModel(meas, chan);
   }

   if (nnps < 6){
     RooFIter iter = ws->components().fwdIterator();
     RooAbsArg *arg;
     while ((arg = iter.next())) {
       if (arg->IsA() == RooRealSumPdf::Class()) {
         arg->setAttribute("BinnedLikelihood");
         std::cout << "component " << arg->GetName() << " is a binned likelihood" << std::endl;
       }
     }
   }
   ws->SetName("BinnedWorkspace");
   ws->writeToFile(name_rootfile);
}

//############## End of Base Algorithms ##############################
//####################################################################
//############## Start Of #Channel tests #############################

static void BM_RooFit_Histfactory(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Minimization);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Eval);
   int bins = state.range(0);
   int cpu = state.range(1);
   TFile *infile = new TFile("HFworkspace.root","RECREATE");
   //   if (infile->IsZombie()) {
   buildBinnedTest(bins, 0, "HFworkspace.root");
   std::cout << "Workspace for tests was created!" << std::endl;
   //}
   infile = TFile::Open("HFworkspace.root");
   RooWorkspace *w = static_cast<RooWorkspace *>(infile->Get("BinnedWorkspace"));
   RooAbsData *data = w->data("obsData");
   ModelConfig *mc = static_cast<ModelConfig *>(w->genobj("ModelConfig"));
   RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());
   RooArgSet * allParams = pdf->getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = pdf->createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setProfile(0);
   m.setLogFile("HFlog");
   RooArgSet * printpars= pdf->getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   for (auto _ : state) {
      m.migrad();
   }
   delete data;
   delete infile;
   delete mc;
   delete pdf;
   delete nll;
}

static void BM_RooFit_HistfactoryNoAtt(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Minimization);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Eval);
   int bins = state.range(0);
   int cpu = state.range(1);
   TFile *infile = new TFile("HFNAworkspace.root","RECREATE");
   //   if (infile->IsZombie()) {
   buildBinnedTest(bins, 7, "HFNAworkspace.root");
   std::cout << "Workspace for tests was created!" << std::endl;
   //}
   infile = TFile::Open("HFNAworkspace.root");
   RooWorkspace *w = static_cast<RooWorkspace *>(infile->Get("BinnedWorkspace"));
   RooAbsData *data = w->data("obsData");
   ModelConfig *mc = static_cast<ModelConfig *>(w->genobj("ModelConfig"));
   RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());
   RooArgSet * allParams = pdf->getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = pdf->createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setProfile(0);
   m.setLogFile("HFlog");
   RooArgSet * printpars= pdf->getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   for (auto _ : state) {
      m.migrad();
   }
   delete data;
   delete infile;
   delete mc;
   delete pdf;
   delete nll;
}

static void BM_RooFit_HistInterp(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Minimization);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Eval);
   int bins = state.range(0);
   int cpu = state.range(1);
   TFile *infile = new TFile("HIworkspace.root","RECREATE");
   //   if (infile->IsZombie()) {
   buildBinnedTest(bins, 1, "HIworkspace.root");
   std::cout << "Workspace for tests was created!" << std::endl;
   //}
   infile = TFile::Open("HIworkspace.root");
   RooWorkspace *w = static_cast<RooWorkspace *>(infile->Get("BinnedWorkspace"));
   RooAbsData *data = w->data("obsData");
   ModelConfig *mc = static_cast<ModelConfig *>(w->genobj("ModelConfig"));
   RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());
   RooArgSet * allParams = pdf->getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = pdf->createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setProfile(0);
   m.setLogFile("HIlog");
   RooArgSet * printpars= pdf->getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   for (auto _ : state) {
      m.migrad();
   }
   delete data;
   delete infile;
   delete mc;
   delete pdf;
   delete nll;
}

static void BM_RooFit_HistNorm(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Minimization);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Eval);
   int bins = state.range(0);
   int cpu = state.range(1);
   TFile *infile = new TFile("HNworkspace.root","RECREATE");
   //   if (infile->IsZombie()) {
   buildBinnedTest(bins, 2, "HNworkspace.root");
   std::cout << "Workspace for tests was created!" << std::endl;
   //}
   infile = TFile::Open("HNworkspace.root");
   RooWorkspace *w = static_cast<RooWorkspace *>(infile->Get("BinnedWorkspace"));
   RooAbsData *data = w->data("obsData");
   ModelConfig *mc = static_cast<ModelConfig *>(w->genobj("ModelConfig"));
   RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());
   RooArgSet * allParams = pdf->getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = pdf->createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setProfile(0);
   m.setLogFile("HNlog");
   RooArgSet * printpars= pdf->getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   for (auto _ : state) {
      m.migrad();
   }
   delete data;
   delete infile;
   delete mc;
   delete pdf;
   delete nll;
}

static void BM_RooFit_HistStatNom(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Minimization);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Eval);
   int bins = state.range(0);
   int cpu = state.range(1);
   TFile *infile = new TFile("HSworkspace.root","RECREATE");
   //   if (infile->IsZombie()) {
   buildBinnedTest(bins, 3, "HSworkspace.root");
   std::cout << "Workspace for tests was created!" << std::endl;
   //}
   infile = TFile::Open("HSworkspace.root");
   RooWorkspace *w = static_cast<RooWorkspace *>(infile->Get("BinnedWorkspace"));
   RooAbsData *data = w->data("obsData");
   ModelConfig *mc = static_cast<ModelConfig *>(w->genobj("ModelConfig"));
   RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());
   RooArgSet * allParams = pdf->getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = pdf->createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setProfile(0);
   m.setLogFile("HNlog");
   RooArgSet * printpars= pdf->getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   for (auto _ : state) {
      m.migrad();
   }
   delete data;
   delete infile;
   delete mc;
   delete pdf;
   delete nll;
}

static void BM_RooFit_HistStatNoAtt(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Minimization);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Eval);
   int bins = state.range(0);
   int cpu = state.range(1);
   TFile *infile = new TFile("HNAworkspace.root","RECREATE");
   //   if (infile->IsZombie()) {
   buildBinnedTest(bins, 6, "HNAworkspace.root");
   std::cout << "Workspace for tests was created!" << std::endl;
   //}
   infile = TFile::Open("HNAworkspace.root");
   RooWorkspace *w = static_cast<RooWorkspace *>(infile->Get("BinnedWorkspace"));
   RooAbsData *data = w->data("obsData");
   ModelConfig *mc = static_cast<ModelConfig *>(w->genobj("ModelConfig"));
   RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());
   RooArgSet * allParams = pdf->getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = pdf->createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setProfile(0);
   m.setLogFile("HNlog");
   RooArgSet * printpars= pdf->getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   for (auto _ : state) {
      m.migrad();
   }
   delete data;
   delete infile;
   delete mc;
   delete pdf;
   delete nll;
}

static void BM_RooFit_ShortStat(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Minimization);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Eval);
   int bins = state.range(0);
   int cpu = state.range(1);
   TFile *infile = new TFile("StatShortworkspace.root","RECREATE");
   //   if (infile->IsZombie()) {
   buildBinnedTest(bins, 3, "StatShortworkspace.root");
   std::cout << "Workspace for tests was created!" << std::endl;
   //}
   infile = TFile::Open("StatShortworkspace.root");
   RooWorkspace *w = static_cast<RooWorkspace *>(infile->Get("BinnedWorkspace"));
   RooAbsData *data = w->data("obsData");
   ModelConfig *mc = static_cast<ModelConfig *>(w->genobj("ModelConfig"));
   RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());
   RooArgSet * allParams = pdf->getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = pdf->createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setProfile(0);
   m.setLogFile("HNlog");
   RooArgSet * printpars= pdf->getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   for (auto _ : state) {
      m.migrad();
   }
   delete data;
   delete infile;
   delete mc;
   delete pdf;
   delete nll;
}


static void BM_RooFit_HistStatHistoSys(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Minimization);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Eval);
   int bins = state.range(0);
   int cpu = state.range(1);
   TFile *infile = new TFile("HSIworkspace.root","RECREATE");
   //   if (infile->IsZombie()) {
   buildBinnedTest(bins, 4, "HSIworkspace.root");
   std::cout << "Workspace for tests was created!" << std::endl;
   //}
   infile = TFile::Open("HSIworkspace.root");
   RooWorkspace *w = static_cast<RooWorkspace *>(infile->Get("BinnedWorkspace"));
   RooAbsData *data = w->data("obsData");
   ModelConfig *mc = static_cast<ModelConfig *>(w->genobj("ModelConfig"));
   RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());
   RooArgSet * allParams = pdf->getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = pdf->createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setProfile(0);
   m.setLogFile("HNlog");
   RooArgSet * printpars= pdf->getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   for (auto _ : state) {
      m.migrad();
   }
   delete data;
   delete infile;
   delete mc;
   delete pdf;
   delete nll;
}

static void BM_RooFit_HistStatTwoChannel(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Minimization);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::NumIntegration);
   RooMsgService::instance().getStream(1).removeTopic(RooFit::Eval);
   int bins = state.range(0);
   int cpu = state.range(1);
   TFile *infile = new TFile("HSCworkspace.root","RECREATE");
   //   if (infile->IsZombie()) {
   buildBinnedTest(bins, 5, "HSCworkspace.root");
   std::cout << "Workspace for tests was created!" << std::endl;
   //}
   infile = TFile::Open("HSCworkspace.root");
   RooWorkspace *w = static_cast<RooWorkspace *>(infile->Get("BinnedWorkspace"));
   RooAbsData *data = w->data("obsData");
   ModelConfig *mc = static_cast<ModelConfig *>(w->genobj("ModelConfig"));
   RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());
   RooArgSet * allParams = pdf->getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = pdf->createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setProfile(0);
   m.setLogFile("HNlog");
   RooArgSet * printpars= pdf->getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   for (auto _ : state) {
      m.migrad();
   }
   delete data;
   delete infile;
   delete mc;
   delete pdf;
   delete nll;
}

static void BM_RooFit_ASUM(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   int bins = state.range(0);
   int cpu = state.range(1);

   RooRealVar x("x","x",0,bins) ;
   x.setBins(bins) ;
   // Parameters
   RooRealVar a0("a0","a0",0) ;
   RooRealVar a1("a1","a1",1,0,2) ;
   RooRealVar a2("a2","a2",0);

   RooPolynomial p0("p0","p0",x);
   RooPolynomial p1("p1","p1",x,RooArgList(a0,a1,a2),0);

   RooDataHist *dh_bkg = p0.generateBinned(x, 1000000000);
   RooDataHist *dh_sig = p1.generateBinned(x, 100000000);
   dh_bkg->SetName("dh_bkg");
   dh_sig->SetName("dh_sig");

   RooWorkspace w = RooWorkspace("w");
   w.import(x);
   w.import(*dh_sig);
   w.import(*dh_bkg);
   w.factory("HistFunc::hf_sig(x,dh_sig)");
   w.factory("HistFunc::hf_bkg(x,dh_bkg)");
   w.factory("ASUM::model(mu[1,0,5]hf_sig,nu[1]*hf_bkg)");

   RooFIter iter = w.components().fwdIterator();
   RooAbsArg *arg;
   while ((arg = iter.next())) {
      if (arg->IsA() == RooRealSumPdf::Class()) {
         arg->setAttribute("BinnedLikelihood");
         std::cout << "component " << arg->GetName() << " is a binned likelihood" << std::endl;
      }
   }

   RooDataHist *data = w.pdf("model")->generateBinned(x, 1100000);
   w.import(*data);
   w.SetName("ASUMWorkspace");
   w.writeToFile("ASUMworkspace.root");

   // Create NLL
   RooAbsPdf * pdf = w.pdf("model");
   RooArgSet * allParams = pdf->getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = pdf->createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setLogFile("asumlog");
   RooArgSet * printpars= pdf->getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   while (state.KeepRunning()) {
      m.migrad();
   }
   delete data;
   delete nll;
}

static void BM_RooFit_ASUMNoAtt(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   int bins = state.range(0);
   int cpu = state.range(1);

   RooRealVar x("x","x",0,bins) ;
   x.setBins(bins) ;
   // Parameters
   RooRealVar a0("a0","a0",0) ;
   RooRealVar a1("a1","a1",1,0,2) ;
   RooRealVar a2("a2","a2",0);

   RooPolynomial p0("p0","p0",x);
   RooPolynomial p1("p1","p1",x,RooArgList(a0,a1,a2),0);

   RooDataHist *dh_bkg = p0.generateBinned(x, 1000000000);
   RooDataHist *dh_sig = p1.generateBinned(x, 100000000);
   dh_bkg->SetName("dh_bkg");
   dh_sig->SetName("dh_sig");

   RooWorkspace w = RooWorkspace("w");
   w.import(x);
   w.import(*dh_sig);
   w.import(*dh_bkg);
   w.factory("HistFunc::hf_sig(x,dh_sig)");
   w.factory("HistFunc::hf_bkg(x,dh_bkg)");
   w.factory("ASUM::model(mu[1,0,5]hf_sig,nu[1]*hf_bkg)");

   RooDataHist *data = w.pdf("model")->generateBinned(x, 1100000);
   w.import(*data);
   w.SetName("ASUMWorkspace");
   w.writeToFile("ASUMworkspace.root");

   // Create NLL
   RooAbsPdf * pdf = w.pdf("model");
   RooArgSet * allParams = pdf->getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = pdf->createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setLogFile("asumlog");
   RooArgSet * printpars= pdf->getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   while (state.KeepRunning()) {
      m.migrad();
   }
   delete data;
   delete nll;
}

static void BM_RooFit_Interp(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   int bins = state.range(0);
   int cpu = state.range(1);

   RooRealVar x("x","x",0,bins) ;
   x.setBins(bins) ;
   // Parameters
   RooRealVar a0("a0","a0",0) ;
   RooRealVar a1("a1","a1",1,0,2) ;
   RooRealVar a2("a2","a2",0);

   RooPolynomial p0("p0","p0",x);
   RooPolynomial p1("p1","p1",x,RooArgList(a0,a1,a2),0);

   RooDataHist *dh_bkg = p0.generateBinned(x, 1000000000);
   RooDataHist *dh_sig = p1.generateBinned(x, 100000000);
   dh_bkg->SetName("dh_bkg");
   dh_sig->SetName("dh_sig");

   a1.setVal(2);
   RooDataHist *dh_sig_up = p1.generateBinned(x, 1100000000);
   dh_sig_up->SetName("dh_sig_up");
   a1.setVal(.5);
   RooDataHist *dh_sig_down = p1.generateBinned(x, 900000000);
   dh_sig_down->SetName("dh_sig_down");


   RooWorkspace w = RooWorkspace("w");
   w.import(x);
   w.import(*dh_sig);
   w.import(*dh_bkg);
   w.import(*dh_sig_up);
   w.import(*dh_sig_down);
   w.factory("HistFunc::hf_sig(x,dh_sig)");
   w.factory("HistFunc::hf_bkg(x,dh_bkg)");
   w.factory("HistFunc::hf_sig_up(x,dh_sig_up)");
   w.factory("HistFunc::hf_sig_down(x,dh_sig_down)");
   w.factory("PiecewiseInterpolation::pi_sig(hf_sig,hf_sig_down,hf_sig_up,alpha[-5,5])");

   //   w.function("pi_sig")->setPositiveDefinite(kTRUE);
   //   w.function("pi_sig")->setAllInterpCodes(4);

   w.factory("ASUM::model(mu[1,0,5]*pi_sig,nu[1]*hf_bkg)");
   w.factory("Gaussian::constraint(alpha,0,1)");
   w.factory("PROD::model2(model,constraint)");

   RooFIter iter = w.components().fwdIterator();
   RooAbsArg *arg;
   while ((arg = iter.next())) {
      if (arg->IsA() == RooRealSumPdf::Class()) {
         arg->setAttribute("BinnedLikelihood");
         std::cout << "component " << arg->GetName() << " is a binned likelihood" << std::endl;
      }
   }

   RooDataHist *data = w.pdf("model2")->generateBinned(x, 1100000);
   w.import(*data);
   w.SetName("Interpworkspace");
   w.writeToFile("Interpworkspace.root");

   // Create NLL
   RooAbsPdf * pdf = w.pdf("model2");
   RooArgSet * allParams = pdf->getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = pdf->createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setLogFile("interplog");
   RooArgSet * printpars= pdf->getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   while (state.KeepRunning()) {
      m.migrad();
   }
   delete data;
   delete nll;
}

static void BM_RooFit_BBlite(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   int bins = state.range(0);
   int cpu = state.range(1);

   RooRealVar x("x","x",0,bins) ;
   x.setBins(bins) ;
   // Parameters
   RooRealVar a0("a0","a0",0) ;
   RooRealVar a1("a1","a1",1,0,2) ;
   RooRealVar a2("a2","a2",0);

   RooPolynomial p0("p0","p0",x);
   RooPolynomial p1("p1","p1",x,RooArgList(a0,a1,a2),0);

   RooDataHist *dh_bkg = p0.generateBinned(x, 100000);
   RooDataHist *dh_sig = p1.generateBinned(x, 10000);
   dh_bkg->SetName("dh_bkg");
   dh_sig->SetName("dh_sig");
   RooParamHistFunc p_ph_sig("p_ph_sig","p_ph_sig",*dh_sig) ;
   RooParamHistFunc p_ph_bkg("p_ph_bkg","p_ph_bkg",*dh_bkg) ; 
   RooParamHistFunc p_ph_bkg2("p_ph_bkg","p_ph_bkg",*dh_bkg,p_ph_sig,1) ; 
   RooRealVar mu("mu","mu",1,0.01,10) ;
   RooRealVar nu("nu","nu",1) ;
   RooRealSumPdf model2_tmp("sp_ph","sp_ph",RooArgList(p_ph_sig,p_ph_bkg2),RooArgList(mu,nu),kTRUE) ;
   RooHistConstraint hc_sigbkg("hc_sigbkg","hc_sigbkg",RooArgSet(p_ph_sig,p_ph_bkg2)) ;
   RooProdPdf model2("model2","model2",hc_sigbkg,Conditional(model2_tmp,x)) ;

   RooWorkspace w = RooWorkspace("w");
   w.import(x);
   w.import(*dh_sig);
   w.import(*dh_bkg);
   w.import(model2);

   RooFIter iter = w.components().fwdIterator();
   RooAbsArg *arg;
   while ((arg = iter.next())) {
      if (arg->IsA() == RooRealSumPdf::Class()) {
         arg->setAttribute("BinnedLikelihood");
         std::cout << "component " << arg->GetName() << " is a binned likelihood" << std::endl;
      }
   }

   RooDataHist *data = model2.generateBinned(x, 1100);
   w.import(*data);
   w.SetName("BBworkspace");
   w.writeToFile("BBworkspace.root");

   // Create NLL
   RooArgSet * allParams = model2.getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = model2.createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setLogFile("interplog");
   RooArgSet * printpars= model2.getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   while (state.KeepRunning()){
      m.migrad();
   }
   delete data;
   delete nll;
}

static void BM_RooFit_BBliteNoAtt(benchmark::State &state)
{
   gErrorIgnoreLevel = kInfo;
   int bins = state.range(0);
   int cpu = state.range(1);

   RooRealVar x("x","x",0,bins) ;
   x.setBins(bins) ;
   // Parameters
   RooRealVar a0("a0","a0",0) ;
   RooRealVar a1("a1","a1",1,0,2) ;
   RooRealVar a2("a2","a2",0);

   RooPolynomial p0("p0","p0",x);
   RooPolynomial p1("p1","p1",x,RooArgList(a0,a1,a2),0);

   RooDataHist *dh_bkg = p0.generateBinned(x, 100000);
   RooDataHist *dh_sig = p1.generateBinned(x, 10000);
   dh_bkg->SetName("dh_bkg");
   dh_sig->SetName("dh_sig");
   RooParamHistFunc p_ph_sig("p_ph_sig","p_ph_sig",*dh_sig) ;
   RooParamHistFunc p_ph_bkg("p_ph_bkg","p_ph_bkg",*dh_bkg) ; 
   RooParamHistFunc p_ph_bkg2("p_ph_bkg","p_ph_bkg",*dh_bkg,p_ph_sig,1) ; 
   RooRealVar mu("mu","mu",1,0.01,10) ;
   RooRealVar nu("nu","nu",1) ;
   RooRealSumPdf model2_tmp("sp_ph","sp_ph",RooArgList(p_ph_sig,p_ph_bkg2),RooArgList(mu,nu),kTRUE) ;
   RooHistConstraint hc_sigbkg("hc_sigbkg","hc_sigbkg",RooArgSet(p_ph_sig,p_ph_bkg2)) ;
   RooProdPdf model2("model2","model2",hc_sigbkg,Conditional(model2_tmp,x)) ;

   RooWorkspace w = RooWorkspace("w");
   w.import(x);
   w.import(*dh_sig);
   w.import(*dh_bkg);
   w.import(model2);

   RooDataHist *data = model2.generateBinned(x, 1100);
   w.import(*data);
   w.SetName("BBNAworkspace");
   w.writeToFile("BBNAworkspace.root");

   // Create NLL
   RooArgSet * allParams = model2.getParameters(data);
   RemoveConstantParameters(allParams);
   RooAbsReal *nll = model2.createNLL(*data, NumCPU(cpu, 0));
   RooMinimizer m(*nll);
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.setLogFile("interplog");
   RooArgSet * printpars= model2.getParameters(data);
   std::cout<<"How many parameters"<<std::endl;
   printpars->Print();
   while (state.KeepRunning()){
      m.migrad();
   }
   delete data;
   delete nll;
}


static void BinArguments(benchmark::internal::Benchmark* b) {
  for (int i = 1; i <=10; ++i )
    for (int j = 1; j <= 3; ++j)
      b->Args({i*10000, j});
}

// BENCHMARK(BM_RooFit_ASUM)->Apply(BinArguments)->UseRealTime()->Iterations(12);
// BENCHMARK(BM_RooFit_ASUMNoAtt)->Apply(BinArguments)->UseRealTime()->Iterations(12);
// BENCHMARK(BM_RooFit_Interp)->Apply(BinArguments)->UseRealTime()->Iterations(12);
// BENCHMARK(BM_RooFit_Histfactory)->Apply(BinArguments)->UseRealTime()->Iterations(12);
// BENCHMARK(BM_RooFit_HistfactoryNoAtt)->Apply(BinArguments)->UseRealTime()->Iterations(12);
// BENCHMARK(BM_RooFit_HistInterp)->Apply(BinArguments)->UseRealTime()->Iterations(12);
// BENCHMARK(BM_RooFit_HistNorm)->Apply(BinArguments)->UseRealTime()->Iterations(12);

static void SmallArguments(benchmark::internal::Benchmark* b) {
  for (int i = 1; i <=10; ++i )
    for (int j = 1; j <= 3; ++j)
      b->Args({i*10, j});
}

BENCHMARK(BM_RooFit_BBlite)->Apply(SmallArguments)->UseRealTime()->Iterations(120);
// BENCHMARK(BM_RooFit_BBliteNoAtt)->Apply(SmallArguments)->UseRealTime()->Iterations(12);
BENCHMARK(BM_RooFit_HistStatNom)->Apply(SmallArguments)->UseRealTime()->Iterations(12);
// BENCHMARK(BM_RooFit_HistStatNoAtt)->Apply(SmallArguments)->UseRealTime()->Iterations(12);
// BENCHMARK(BM_RooFit_HistStatHistoSys)->Apply(SmallArguments)->UseRealTime()->Iterations(12);
// BENCHMARK(BM_RooFit_HistStatTwoChannel)->Apply(SmallArguments)->UseRealTime()->Iterations(12);

static void TinyArguments(benchmark::internal::Benchmark* b) {
  for (int i = 2; i <=20; ++i )
    for (int j = 1; j <= 3; ++j)
      b->Args({i, j});
}

//BENCHMARK(BM_RooFit_ShortStat)->Apply(TinyArguments)->UseRealTime()->Iterations(12);

BENCHMARK_MAIN();
