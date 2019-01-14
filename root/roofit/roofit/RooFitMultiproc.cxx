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
#include <MultiProcess/NLLVar.h>
#include <MultiProcess/GradMinimizer.h>
#include "RooAddPdf.h"

#include <ROOT/RMakeUnique.hxx>  // make_unique
#include "RooRandom.h"

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

void buildBinnedTest(int n_channels = 1, int nbins = 10, int nnps = 1, const char *name_rootfile = "", bool verbose = false)
{
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
         if (verbose) std::cout << "component " << arg->GetName() << " is a binned likelihood\n";
      }
   }
   ws->SetName("BinnedWorkspace");
   ws->writeToFile(name_rootfile);
}


std::tuple<std::unique_ptr<RooAbsReal>, std::unique_ptr<RooArgSet>>
generate_1D_gaussian_pdf_nll(RooWorkspace &w, unsigned long N_events) {
  w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");

  RooAbsPdf *pdf = w.pdf("g");
  RooRealVar *mu = w.var("mu");

  RooDataSet *data = pdf->generate(RooArgSet(*w.var("x")), N_events);
  mu->setVal(-2.9);

  std::unique_ptr<RooAbsReal> nll {pdf->createNLL(*data)};

  // save initial values for the start of all minimizations
  std::unique_ptr<RooArgSet> values = std::make_unique<RooArgSet>(*mu, *pdf, *nll, "values");

  return std::make_tuple(std::move(nll), std::move(values));
}

// return two unique_ptrs, the first because nll is a pointer,
// the second because RooArgSet doesn't have a move ctor
std::tuple<std::unique_ptr<RooAbsReal>, std::unique_ptr<RooArgSet>>
generate_ND_gaussian_pdf_nll(RooWorkspace &w, unsigned int n, unsigned long N_events) {
  RooArgSet obs_set;

  // create gaussian parameters
  double mean[n], sigma[n];
  for (unsigned ix = 0; ix < n; ++ix) {
    mean[ix] = RooRandom::randomGenerator()->Gaus(0, 2);
    sigma[ix] = 0.1 + abs(RooRandom::randomGenerator()->Gaus(0, 2));
  }

  // create gaussians and also the observables and parameters they depend on
  for (unsigned ix = 0; ix < n; ++ix) {
    std::ostringstream os;
    os << "Gaussian::g" << ix
       << "(x" << ix << "[-10,10],"
       << "m" << ix << "[" << mean[ix] << ",-10,10],"
       << "s" << ix << "[" << sigma[ix] << ",0.1,10])";
    w.factory(os.str().c_str());
  }

  // create uniform background signals on each observable
  for (unsigned ix = 0; ix < n; ++ix) {
    {
      std::ostringstream os;
      os << "Uniform::u" << ix << "(x" << ix << ")";
      w.factory(os.str().c_str());
    }

    // gather the observables in a list for data generation below
    {
      std::ostringstream os;
      os << "x" << ix;
      obs_set.add(*w.arg(os.str().c_str()));
    }
  }

  RooArgSet pdf_set = w.allPdfs();

  // create event counts for all pdfs
  RooArgSet count_set;

  // ... for the gaussians
  for (unsigned ix = 0; ix < n; ++ix) {
    std::stringstream os, os2;
    os << "Nsig" << ix;
    os2 << "#signal events comp " << ix;
    RooRealVar a(os.str().c_str(), os2.str().c_str(), N_events/10, 0., 10*N_events);
    w.import(a);
    // gather in count_set
    count_set.add(*w.arg(os.str().c_str()));
  }
  // ... and for the uniform background components
  for (unsigned ix = 0; ix < n; ++ix) {
    std::stringstream os, os2;
    os << "Nbkg" << ix;
    os2 << "#background events comp " << ix;
    RooRealVar a(os.str().c_str(), os2.str().c_str(), N_events/10, 0., 10*N_events);
    w.import(a);
    // gather in count_set
    count_set.add(*w.arg(os.str().c_str()));
  }

  RooAddPdf* sum = new RooAddPdf("sum", "gaussians+uniforms", pdf_set, count_set);
  w.import(*sum);  // keep sum around after returning

  // --- Generate a toyMC sample from composite PDF ---
  RooDataSet *data = sum->generate(obs_set, N_events);

  std::unique_ptr<RooAbsReal> nll {sum->createNLL(*data)};

  // set values randomly so that they actually need to do some fitting
  for (unsigned ix = 0; ix < n; ++ix) {
    {
      std::ostringstream os;
      os << "m" << ix;
      dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->setVal(RooRandom::randomGenerator()->Gaus(0, 2));
    }
    {
      std::ostringstream os;
      os << "s" << ix;
      dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->setVal(0.1 + abs(RooRandom::randomGenerator()->Gaus(0, 2)));
    }
  }

  // gather all values of parameters, pdfs and nll here for easy
  // saving and restoring
  std::unique_ptr<RooArgSet> all_values = std::make_unique<RooArgSet>(pdf_set, count_set, "all_values");
  all_values->add(*nll);
  all_values->add(*sum);
  for (unsigned ix = 0; ix < n; ++ix) {
    {
      std::ostringstream os;
      os << "m" << ix;
      all_values->add(*w.arg(os.str().c_str()));
    }
    {
      std::ostringstream os;
      os << "s" << ix;
      all_values->add(*w.arg(os.str().c_str()));
    }
  }

  return std::make_tuple(std::move(nll), std::move(all_values));
}

//############## End of Base Algorithms ##############################
//####################################################################
//############## Start Of #Channel tests #############################


static void BM_RooFit_BinnedMultiProcGradient(benchmark::State &state) {
  RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

  int bins = state.range(0);
  auto NumCPU = static_cast<std::size_t>(state.range(1));

  buildBinnedTest(4, bins, 0, "workspace.root", true);

  TFile * infile = TFile::Open("workspace.root");
  auto w = dynamic_cast<RooWorkspace *>(infile->Get("BinnedWorkspace"));
  RooAbsData *data = w->data("obsData");
  auto mc = dynamic_cast<ModelConfig *>(w->genobj("ModelConfig"));
  RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());

  RooAbsReal *nll = pdf->createNLL(*data);

  MultiProcess::GradMinimizer m(*nll, NumCPU);

  m.setPrintLevel(-1);
  m.setStrategy(0);
  m.setProfile(false);

  RooArgSet* values = nll->getParameters(data);
  values->add(*pdf);
  values->add(*nll);
//  std::unique_ptr<RooArgSet> savedValues = std::make_unique<RooArgSet>(values.snapshot());
  std::unique_ptr<RooArgSet> savedValues {dynamic_cast<RooArgSet *>(values->snapshot())};

  for (auto _ : state) {
    // reset values
    *values = *savedValues;

    auto start = std::chrono::high_resolution_clock::now();

    // do minimization
    m.migrad();
    MultiProcess::TaskManager::instance()->terminate();

    // report time
    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  delete values;
  delete nll;
  delete data;
  delete infile;
}


static void BM_RooFit_1DUnbinnedGaussianMultiProcessGradMinimizer(benchmark::State &state) {
  RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

  auto NumCPU = static_cast<std::size_t>(state.range(0));

  std::size_t seed = 1;

  RooRandom::randomGenerator()->SetSeed(seed);

  RooWorkspace w = RooWorkspace();

  std::unique_ptr<RooAbsReal> nll;
  std::unique_ptr<RooArgSet> values;
  std::tie(nll, values) = generate_1D_gaussian_pdf_nll(w, 800000);

  MultiProcess::GradMinimizer m(*nll, NumCPU);

  m.setPrintLevel(-1);
  m.setStrategy(0);
  m.setProfile(false);

//  RooArgSet *savedValues = dynamic_cast<RooArgSet *>(values->snapshot());
  std::unique_ptr<RooArgSet> savedValues {dynamic_cast<RooArgSet *>(values->snapshot())};

  for (auto _ : state) {
    // reset values
    *values = *savedValues;

    auto start = std::chrono::high_resolution_clock::now();

    // do minimization
    m.migrad();
    MultiProcess::TaskManager::instance()->terminate();

    // report time
    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}


static void BM_RooFit_NDUnbinnedGaussianMultiProcessGradMinimizer(benchmark::State &state) {
  RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

  auto NumCPU = static_cast<std::size_t>(state.range(0));
  auto n_dim = static_cast<std::size_t>(state.range(1));

  std::size_t seed = 1;

  RooRandom::randomGenerator()->SetSeed(seed);

  RooWorkspace w = RooWorkspace();

  std::unique_ptr<RooAbsReal> nll;
  std::unique_ptr<RooArgSet> values;
  std::tie(nll, values) = generate_ND_gaussian_pdf_nll(w, n_dim, 100000/n_dim);

  MultiProcess::GradMinimizer m(*nll, NumCPU);

  m.setPrintLevel(-1);
  m.setStrategy(0);
  m.setProfile(false);

//  RooArgSet *savedValues = dynamic_cast<RooArgSet *>(values->snapshot());
  std::unique_ptr<RooArgSet> savedValues {dynamic_cast<RooArgSet *>(values->snapshot())};

  for (auto _ : state) {
    // reset values
    *values = *savedValues;

    auto start = std::chrono::high_resolution_clock::now();

    // do minimization
    m.migrad();
    MultiProcess::TaskManager::instance()->terminate();

    // report time
    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}


static void BM_RooFit_MP_GradMinimizer_workspace_file(benchmark::State &state) {
  RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

  auto NumCPU = static_cast<std::size_t>(state.range(0));

  std::size_t seed = 1;
  RooRandom::randomGenerator()->SetSeed(seed);

  TFile *_file0 = TFile::Open("bench_this_workspace.root");

  // TODO: add functionality to read parameters from text file (names of workspace, dataset, modelconfig)

  RooWorkspace* w = static_cast<RooWorkspace*>(gDirectory->Get("HWWRun2GGF"));

  RooAbsData *data = w->data("obsData");
  auto mc = dynamic_cast<ModelConfig *>(w->genobj("ModelConfig"));
  RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());

  RooAbsReal *nll = pdf->createNLL(*data);

  RooArgSet* values = nll->getParameters(data);
  values->add(*pdf);
  values->add(*nll);

  MultiProcess::GradMinimizer m(*nll, NumCPU);

  m.setPrintLevel(-1);
  m.setStrategy(0);
  m.setProfile(false);

  std::unique_ptr<RooArgSet> savedValues {dynamic_cast<RooArgSet *>(values->snapshot())};

  for (auto _ : state) {
    // reset values
    *values = *savedValues;

    auto start = std::chrono::high_resolution_clock::now();

    // do minimization
    m.migrad();
    MultiProcess::TaskManager::instance()->terminate();

    // report time
    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}


static void BinArguments(benchmark::internal::Benchmark* b) {
  for (int bins = 6; bins <= 15; bins+=3) {
    for (int cpus = 1; cpus <= 8; ++cpus) {
      b->Args({bins, cpus});
    }
  }
}
BENCHMARK(BM_RooFit_BinnedMultiProcGradient)->Apply(BinArguments)->UseManualTime()->Unit(benchmark::kMillisecond);

static void NumCPUArguments(benchmark::internal::Benchmark* b) {
  for (int cpus = 1; cpus <= 8; ++cpus) {
    b->Args({cpus});
  }
}
BENCHMARK(BM_RooFit_1DUnbinnedGaussianMultiProcessGradMinimizer)->Apply(NumCPUArguments)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RooFit_MP_GradMinimizer_workspace_file)->Apply(NumCPUArguments)->UseManualTime()->Unit(benchmark::kMillisecond);

static void CPUsDimsArguments(benchmark::internal::Benchmark* b) {
  for (int cpus = 1; cpus <= 8; ++cpus) {
    for (int dims = 1; dims <= 8; ++dims) {
      b->Args({cpus, dims});
    }
  }
}
BENCHMARK(BM_RooFit_NDUnbinnedGaussianMultiProcessGradMinimizer)->Apply(CPUsDimsArguments)->UseManualTime()->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
