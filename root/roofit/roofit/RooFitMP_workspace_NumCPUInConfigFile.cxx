#include <fstream>  // ifstream
#include <sstream>  // stringstream
#include <chrono>
#include <stdexcept>  // runtime_error

#include "TFile.h"
#include "TDirectory.h"

#include "RooMsgService.h"
#include "RooRandom.h"
#include "RooWorkspace.h"
#include "MultiProcess/GradMinimizer.h"
#include "RooTimer.h"

#include "RooStats/ModelConfig.h"

#include "benchmark/benchmark.h"

std::tuple<std::string, std::string, std::string, std::string, std::size_t> read_config_file() {
  std::string filename, workspace_name, dataset_name, modelconfig_name, NumCPU_s;
  std::size_t NumCPU;

  std::ifstream workspace_file_config_file("workspace_benchmark.conf");
  if (workspace_file_config_file.is_open()) {
    std::getline(workspace_file_config_file, filename);
    std::getline(workspace_file_config_file, workspace_name);
    std::getline(workspace_file_config_file, dataset_name);
    std::getline(workspace_file_config_file, modelconfig_name);
    std::getline(workspace_file_config_file, NumCPU_s);
    std::stringstream ss;
    ss << NumCPU_s;
    ss >> NumCPU;
  } else {
    throw runtime_error("Could not open workspace_benchmark.conf configuration file");
  }
  return {filename, workspace_name, dataset_name, modelconfig_name, NumCPU};
}


RooAbsReal * create_nll(RooAbsPdf * pdf, RooAbsData * data,
                        const RooArgSet * global_observables,
                        const RooArgSet * nuisance_parameters) {
  RooAbsReal *nll;

  if (global_observables != nullptr) {
    nll = pdf->createNLL(*data,
                         RooFit::GlobalObservables(*global_observables),
                         RooFit::Constrain(*nuisance_parameters),
                         RooFit::Offset(kTRUE));
  } else {
    nll = pdf->createNLL(*data,
                         RooFit::Constrain(*nuisance_parameters),
                         RooFit::Offset(kTRUE));
  }

  return nll;
}


static void BM_RooFit_MP_GradMinimizer_workspace_file_NumCPUInConfigFile(benchmark::State &state) {
//  RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);
  RooMsgService::instance().deleteStream(0);
  RooMsgService::instance().deleteStream(0);

  RooMsgService::instance().addStream(RooFit::DEBUG, RooFit::Topic(RooFit::Benchmarking1));
  RooMsgService::instance().addStream(RooFit::DEBUG, RooFit::Topic(RooFit::Benchmarking2));

  std::size_t seed = 1;
  RooRandom::randomGenerator()->SetSeed(seed);

  // read filename and dataset / modelconfig names from configuration file
  std::string filename, workspace_name, dataset_name, modelconfig_name;
  std::size_t NumCPU;
  Bool_t offset;
  std::tie(filename, workspace_name, dataset_name, modelconfig_name, NumCPU) = read_config_file();

  TFile *_file0 = TFile::Open(filename.c_str());
  RooWorkspace* w = static_cast<RooWorkspace*>(gDirectory->Get(workspace_name.c_str()));

  RooAbsData *data = w->data(dataset_name.c_str());
  auto mc = dynamic_cast<RooStats::ModelConfig *>(w->genobj(modelconfig_name.c_str()));

  auto global_observables = mc->GetGlobalObservables();
  auto nuisance_parameters = mc->GetNuisanceParameters();

  RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());

  RooAbsReal *nll = create_nll(pdf, data, global_observables, nuisance_parameters);

  RooArgSet* values = nll->getParameters(data);
  values->add(*pdf);
  values->add(*nll);

  RooFit::MultiProcess::GradMinimizer m(*nll, NumCPU);

  m.setPrintLevel(-1);
  m.setStrategy(0);
  m.setProfile(false);
  m.optimizeConst(2);
  m.setMinimizerType("Minuit2");

  std::unique_ptr<RooArgSet> savedValues {dynamic_cast<RooArgSet *>(values->snapshot())};

  RooWallTimer timer;

  auto get_time = [](){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();};

  for (auto _ : state) {
    // reset values
    *values = *savedValues;

    auto start = std::chrono::high_resolution_clock::now();

    // do minimization
    oocxcoutD((TObject*)nullptr,Benchmarking1) << "start migrad at " << get_time() << " with NumCPU = " << NumCPU << std::endl;
    m.migrad();
    oocxcoutD((TObject*)nullptr,Benchmarking1) << "end migrad at " << get_time() << std::endl;
    timer.start();
    RooFit::MultiProcess::TaskManager::instance()->terminate();
    timer.stop();
    oocxcoutD((TObject*)nullptr,Benchmarking1) << "terminate: " << timer.timing_s() << "s" << std::endl;

    // report time
    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  delete values;
  delete nll;
  delete _file0;
}

static void BM_RooFit_RooMinimizer_workspace_file_NumCPUInConfigFile(benchmark::State &state) {
  RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

  std::size_t seed = 1;
  RooRandom::randomGenerator()->SetSeed(seed);

  // read filename and dataset / modelconfig names from configuration file
  std::string filename, workspace_name, dataset_name, modelconfig_name;
  std::size_t NumCPU;  // not used here
  std::tie(filename, workspace_name, dataset_name, modelconfig_name, NumCPU) = read_config_file();

  TFile *_file0 = TFile::Open(filename.c_str());
  RooWorkspace* w = static_cast<RooWorkspace*>(gDirectory->Get(workspace_name.c_str()));

  RooAbsData *data = w->data(dataset_name.c_str());
  auto mc = dynamic_cast<RooStats::ModelConfig *>(w->genobj(modelconfig_name.c_str()));

  auto global_observables = mc->GetGlobalObservables();
  auto nuisance_parameters = mc->GetNuisanceParameters();

  RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());

  RooAbsReal *nll = create_nll(pdf, data, global_observables, nuisance_parameters);

  RooArgSet* values = nll->getParameters(data);
  values->add(*pdf);
  values->add(*nll);

  RooMinimizer m(*nll);

  m.setPrintLevel(-1);
  m.setStrategy(0);
  m.setProfile(false);
  m.optimizeConst(2);
  m.setMinimizerType("Minuit2");

  std::unique_ptr<RooArgSet> savedValues {dynamic_cast<RooArgSet *>(values->snapshot())};

  for (auto _ : state) {
    // reset values
    *values = *savedValues;

    auto start = std::chrono::high_resolution_clock::now();

    // do minimization
    m.migrad();

    // report time
    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  delete values;
  delete nll;
  delete _file0;
}


BENCHMARK(BM_RooFit_MP_GradMinimizer_workspace_file_NumCPUInConfigFile)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RooFit_RooMinimizer_workspace_file_NumCPUInConfigFile)->UseManualTime()->Unit(benchmark::kMillisecond);


BENCHMARK_MAIN();
