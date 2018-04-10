/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   AL, Alfio Lazzaro,   INFN Milan,        alfio.lazzaro@mi.infn.it        *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *   VC, Vince Croft,     DIANA / NYU,        vincent.croft@cern.ch          *
 *                                                                           *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef __ROOFIT_NOROOGRADMINIMIZER

//////////////////////////////////////////////////////////////////////////////
//
// RooGradMinimizerFcn is am interface class to the ROOT::Math function
// for minization. See RooGradMinimizer.cxx for more information.
//                                                                                   

#include <iostream>

#include "RooFit.h"

#include "Riostream.h"

#include "TIterator.h"
#include "TClass.h"

#include "RooAbsArg.h"
#include "RooAbsPdf.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooAbsRealLValue.h"
#include "RooMsgService.h"

//#include "RooMinimizer.h"
#include "RooGradMinimizerFcn.h"
#include "RooGradMinimizer.h"

#include "Fit/Fitter.h"
#include "Math/Minimizer.h"
//#include "Minuit2/MnMachinePrecision.h"

#include <algorithm> // std::equal


RooGradMinimizerFcn::RooGradMinimizerFcn(RooAbsReal *funct, RooGradMinimizer* context,
                                         RooGradientFunction::GradientCalculatorMode grad_mode, bool verbose) :
    RooGradientFunction(funct, grad_mode, verbose),
    _context(context) {
  set_strategy(ROOT::Math::MinimizerOptions::DefaultStrategy());
  set_error_level(ROOT::Math::MinimizerOptions::DefaultErrorDef());
}


RooGradMinimizerFcn::RooGradMinimizerFcn(const RooGradMinimizerFcn& other) :
    RooGradientFunction(other),
    _context(other._context) {}


ROOT::Math::IMultiGradFunction* RooGradMinimizerFcn::Clone() const {
  return new RooGradMinimizerFcn(*this) ;
}


void RooGradMinimizerFcn::BackProp(const ROOT::Fit::FitResult &results)
{
  // Transfer MINUIT fit results back into RooFit objects

  for (unsigned index = 0; index < NDim(); index++) {
    Double_t value = results.Value(index);
    SetPdfParamVal(index, value);

    // Set the parabolic error    
    Double_t err = results.Error(index);
    SetPdfParamErr(index, err);
    
    Double_t eminus = results.LowerError(index);
    Double_t eplus = results.UpperError(index);

    if(eplus > 0 || eminus < 0) {
      // Store the asymmetric error, if it is available
      SetPdfParamErr(index, eminus,eplus);
    } else {
      // Clear the asymmetric error
      ClearPdfParamAsymErr(index) ;
    }
  }

}


void RooGradMinimizerFcn::ApplyCovarianceMatrix(TMatrixDSym& V) 
{
  // Apply results of given external covariance matrix. i.e. propagate its errors
  // to all RRV parameter representations and give this matrix instead of the
  // HESSE matrix at the next save() call

  for (unsigned i = 0 ; i < NDim() ; i++) {
    // Skip fixed parameters
    if (GetFloatParamList()->at(i)->isConstant()) {
      continue ;
    }
    SetPdfParamErr(i, sqrt(V(i,i))) ;		  
  }

}


////////////////////////////////////////////////////////////////////////////////

// it's not actually const, it mutates mutables, but it has to be defined
// const because it's called from DoDerivative, which insists on constness

std::vector<ROOT::Fit::ParameterSettings>& RooGradMinimizerFcn::parameter_settings() const {
  return _context->fitter()->Config().ParamsSettings();
}

void RooGradMinimizerFcn::set_strategy(int istrat) {
  assert (istrat >= 0);
  ROOT::Minuit2::MnStrategy strategy(static_cast<unsigned int>(istrat));

  set_step_tolerance(strategy.GradientStepTolerance());
  set_grad_tolerance(strategy.GradientTolerance());
  set_ncycles(strategy.GradientNCycles());
}

#endif

