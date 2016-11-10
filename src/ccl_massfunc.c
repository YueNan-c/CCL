#include "ccl.h"
#include "ccl_core.h"
#include "ccl_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gsl/gsl_integration.h"
#include "gsl/gsl_interp.h"
#include "gsl/gsl_spline.h"
#include "ccl_power.h"
#include "ccl_massfunc.h"
#include "ccl_error.h"

/*----- ROUTINE: ccl_massfunc_ftinker -----
INPUT: cosmology+parameters, a smoothing scale, and a redshift
TASK: outputs ftinker for calculation in the halo mass function. Assumes
  Tinker 2008 Fitting Function (arxiv 0803.2706 )
*/

static double massfunc_f(ccl_cosmology *cosmo, double halo_mass,double redshift)
{
  double tinker_A, tinker_a, tinker_b, tinker_c;
  double alpha, overdensity_delta;
  double sigma=ccl_sigmaM(cosmo,halo_mass,redshift);

  switch(cosmo->config.mass_function_method){
  case ccl_tinker:
    
    //TODO: maybe use macros for numbers
    overdensity_delta = 200.0;
    tinker_A = 0.186*pow(1+redshift, -0.14);
    tinker_a = 1.47*pow(1+redshift, -0.06);
    alpha = pow(10, -1.0*pow(0.75 / log10(overdensity_delta / 75.0), 1.2 ));
    tinker_b = 2.57*pow(1+redshift, -1.0*alpha);
    tinker_c = 1.19;

    return tinker_A*(pow(sigma/tinker_b,-tinker_a)+1.0)*exp(-tinker_c/sigma/sigma);
    break;
  default:
    cosmo->status = 11;
    sprintf(cosmo->status_message ,
	    "ccl_massfunc.c: ccl_massfunc(): Unknown or non-implemented mass function method: %d \n",
	    cosmo->config.mass_function_method);
    return 0;
  }
}

void ccl_cosmology_compute_sigma(ccl_cosmology * cosmo)
{
    if(cosmo->computed_sigma)
        return;

    // create linearly-spaced values of the halo mass.
    int nm=0;
    double * m = ccl_linear_spacing(LOGM_SPLINE_MIN, LOGM_SPLINE_MAX, LOGM_SPLINE_DELTA, &nm);
    if (m==NULL ||
        (fabs(m[0]-LOGM_SPLINE_MIN)>1e-5) ||
        (fabs(m[nm-1]-LOGM_SPLINE_MAX)>1e-5) ||
        (m[nm-1]>10E17)
        ) {
       cosmo->status =2;
       strcpy(cosmo->status_message,"ccl_cosmology_compute_sigmas(): Error creating linear spacing in m\n");
       return;
    }
    
    // allocate space for y, to be filled with sigma and dlnsigma_dlogm
    double *y = malloc(sizeof(double)*nm);
    double haloradius; 
   
   // fill in sigma
   for (int i=0; i<nm; i++){
     haloradius = ccl_massfunc_m2r(cosmo, pow(10,m[i]));
     y[i] = log10(ccl_sigmaR(cosmo, haloradius));
   }
   gsl_spline * logsigma = gsl_spline_alloc(M_SPLINE_TYPE, nm);
   int status = gsl_spline_init(logsigma, m, y, nm);
   if (status){
     free(m);
     free(y);
     gsl_spline_free(logsigma);
     cosmo->status = 4;
     strcpy(cosmo->status_message, "ccl_massfunc.c: ccl_cosmology_compute_sigma(): Error creating sigma(M) spline\n");
     return;
   }

   for (int i=0; i<nm; i++){
     if(i==0){
       y[i] = log(pow(10, gsl_spline_eval(logsigma, m[i], NULL)))-log(pow(10,gsl_spline_eval(logsigma, m[i]+LOGM_SPLINE_DELTA/2., NULL)));
       y[i] = 2.*y[i] / LOGM_SPLINE_DELTA;
     }
     else if (i==nm-1){
       y[i] = log(pow(10, gsl_spline_eval(logsigma, m[i]-LOGM_SPLINE_DELTA/2., NULL)))-log(pow(10,gsl_spline_eval(logsigma, m[i], NULL)));
       y[i] = 2.*y[i] / LOGM_SPLINE_DELTA;
     }
     else{
       y[i] = (log(pow(10,gsl_spline_eval(logsigma, m[i]-LOGM_SPLINE_DELTA/2., NULL)))-log(pow(10,gsl_spline_eval(logsigma, m[i]+LOGM_SPLINE_DELTA/2., NULL))));
       y[i] = y[i] / LOGM_SPLINE_DELTA;
     }
   }

   gsl_spline * dlnsigma_dlogm = gsl_spline_alloc(M_SPLINE_TYPE, nm);
   status = gsl_spline_init(dlnsigma_dlogm, m, y, nm);
   if (status){
     free(m);
     free(y);
     gsl_spline_free(logsigma);
     cosmo->status = 4;
     strcpy(cosmo->status_message, "ccl_massfunc.c: ccl_cosmology_compute_sigma(): Error creating dlnsigma/dlogM spline\n");
     return;
   }

   if(cosmo->data.accelerator_m==NULL)
     cosmo->data.accelerator_m=gsl_interp_accel_alloc();
   cosmo->data.logsigma = logsigma;
   cosmo->data.dlnsigma_dlogm = dlnsigma_dlogm;
   cosmo->computed_sigma = true;

   free(m);
   free(y);
}

/*----- ROUTINE: ccl_massfunc -----
INPUT: ccl_cosmology * cosmo, ccl_config to decide on which mass func
TASK: return dn/dM according to some methodology
*/
double ccl_massfunc(ccl_cosmology *cosmo, double halo_mass, double redshift)
{
  if (!cosmo->computed_sigma){
    ccl_cosmology_compute_sigma(cosmo);
    ccl_check_status(cosmo);
  }

  double f,deriv,rho_m,logmass;
  
  logmass = log10(halo_mass);
  rho_m = RHO_CRITICAL*cosmo->params.Omega_m*cosmo->params.h*cosmo->params.h;
  f=massfunc_f(cosmo,halo_mass,redshift);
  deriv = gsl_spline_eval(cosmo->data.dlnsigma_dlogm, logmass, cosmo->data.accelerator_m);
  return f;
  //return f*rho_m*deriv/halo_mass;
}

/*---- ROUTINE: ccl_massfunc_m2r -----
INPUT: ccl_cosmology * cosmo, halo_mass in units of Msun/h
TASK: takes smoothing halo mass and converts to smoothing halo radius
  in units of Mpc.
*/
double ccl_massfunc_m2r(ccl_cosmology * cosmo, double halo_mass)
{
    double rho_m, rho_crit, halo_radius;

    //TODO: make this neater
    rho_m = RHO_CRITICAL*cosmo->params.Omega_m;

    halo_mass = halo_mass*cosmo->params.h;
    halo_radius = pow((3.0*halo_mass) / (4*M_PI*rho_m), (1.0/3.0));

    return halo_radius/cosmo->params.h;
}

double ccl_sigmaM(ccl_cosmology * cosmo, double halo_mass, double redshift)
{
    double sigmaM;

    if (!cosmo->computed_sigma){
        ccl_cosmology_compute_sigma(cosmo);
        ccl_check_status(cosmo);
    }

    sigmaM = pow(10,gsl_spline_eval(cosmo->data.logsigma, log10(halo_mass), cosmo->data.accelerator_m));
    sigmaM = sigmaM*ccl_growth_factor(cosmo, 1.0/(1.0+redshift));

    return sigmaM;
}
