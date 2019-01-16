#include <fstream>  // ifstream
#include <chrono>
#include <stdexcept>  // runtime_error

#include "TFile.h"
#include "TDirectory.h"

#include "RooMsgService.h"
#include "RooRandom.h"
#include "RooWorkspace.h"
#include "MultiProcess/GradMinimizer.h"

#include "RooStats/ModelConfig.h"

#include "benchmark/benchmark.h"


static void BM_RooFit_MP_GradMinimizer_workspace_file(benchmark::State &state) {
  RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

  auto NumCPU = static_cast<std::size_t>(state.range(0));

  std::size_t seed = 1;
  RooRandom::randomGenerator()->SetSeed(seed);

  // read filename and dataset / modelconfig names from configuration file
  std::string filename, workspace_name, dataset_name, modelconfig_name;
  std::ifstream workspace_file_config_file("workspace_benchmark.conf");
  if (workspace_file_config_file.is_open()) {
    std::getline(workspace_file_config_file, filename);
    std::getline(workspace_file_config_file, workspace_name);
    std::getline(workspace_file_config_file, dataset_name);
    std::getline(workspace_file_config_file, modelconfig_name);
  } else {
    throw runtime_error("Could not open workspace_benchmark.conf configuration file");
  }

  TFile *_file0 = TFile::Open(filename.c_str());
  RooWorkspace* w = static_cast<RooWorkspace*>(gDirectory->Get(workspace_name.c_str()));

  RooAbsData *data = w->data(dataset_name.c_str());
  auto mc = dynamic_cast<RooStats::ModelConfig *>(w->genobj(modelconfig_name.c_str()));
  RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());

  RooAbsReal *nll = pdf->createNLL(*data);

  RooArgSet* values = nll->getParameters(data);
  values->add(*pdf);
  values->add(*nll);

  RooFit::MultiProcess::GradMinimizer m(*nll, NumCPU);

  m.setPrintLevel(-1);
  m.setStrategy(0);
  m.setProfile(false);

  std::unique_ptr<RooArgSet> savedValues {dynamic_cast<RooArgSet *>(values->snapshot())};

  for (auto _ : state) {
    // reset values
    *values = *savedValues;

    auto start = std::chrono::high_resolution_clock::now();

    // do minimization
    std::cout << "start migrad\n";
    m.migrad();
    std::cout << "end migrad\n";
    RooFit::MultiProcess::TaskManager::instance()->terminate();

    // report time
    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

static void BM_RooFit_RooMinimizer_workspace_file(benchmark::State &state) {
  RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR);

  std::size_t seed = 1;
  RooRandom::randomGenerator()->SetSeed(seed);

  // read filename and dataset / modelconfig names from configuration file
  std::string filename, workspace_name, dataset_name, modelconfig_name;
  std::ifstream workspace_file_config_file("workspace_benchmark.conf");
  if (workspace_file_config_file.is_open()) {
    std::getline(workspace_file_config_file, filename);
    std::getline(workspace_file_config_file, workspace_name);
    std::getline(workspace_file_config_file, dataset_name);
    std::getline(workspace_file_config_file, modelconfig_name);
  } else {
    throw runtime_error("Could not open workspace_benchmark.conf configuration file");
  }

  TFile *_file0 = TFile::Open(filename.c_str());
  RooWorkspace* w = static_cast<RooWorkspace*>(gDirectory->Get(workspace_name.c_str()));

  RooAbsData *data = w->data(dataset_name.c_str());
  auto mc = dynamic_cast<RooStats::ModelConfig *>(w->genobj(modelconfig_name.c_str()));
  RooAbsPdf *pdf = w->pdf(mc->GetPdf()->GetName());

  RooAbsReal *nll = pdf->createNLL(*data);

  RooArgSet* values = nll->getParameters(data);
  values->add(*pdf);
  values->add(*nll);

  RooMinimizer m(*nll);

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

    // report time
    auto end   = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}


static void NumCPUArguments(benchmark::internal::Benchmark* b) {
  for (int cpus = 1; cpus <= 8; ++cpus) {
    b->Args({cpus});
  }
}

BENCHMARK(BM_RooFit_MP_GradMinimizer_workspace_file)->Apply(NumCPUArguments)->UseManualTime()->Unit(benchmark::kMillisecond);
BENCHMARK(BM_RooFit_RooMinimizer_workspace_file)->UseManualTime()->Unit(benchmark::kMillisecond);


BENCHMARK_MAIN();
