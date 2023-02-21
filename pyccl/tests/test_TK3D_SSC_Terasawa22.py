import pyccl as ccl
import numpy as np
#import TK3D_SSC_Terasawa22 from tk3d

def test_tk3d_ssc_terasawa22():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.7, sigma8=0.83,
                           n_s=0.96, transfer_function='boltzmann_camb', matter_power_spectrum='camb')
    a_arr = np.array([0.5, 1.0])
    lk_arr = np.linspace(-5, 5, 10)
    #tkk_arr = np.ones((2, 10, 10))
    
    # Create Tk3D object
    tk3dssc = ccl.tk3d.Tk3D_SSC_Terasawa22(cosmo=cosmo,
                                            lk_arr=lk_arr,
                                            a_arr=a_arr,
                                            extrap_order_lok=1,
                                            extrap_order_hik=1,
                                            use_log=False,
                                            deltah=0.02)
    
    # Evaluate trispectrum
   # k1 = 0.1
  #  k2 = 0.2
  #  a = 0.7
  #  trisp = tk3d.TK3D_ssc_terasawa22(k1, k2, a)
    
    # Check result
    assert np.all(np.isfinite(tk3dssc))
    #assert np.isclose(tk3dssc, 1.0, rtol=1e-5)



# import pyccl
# import pytest
# #from pyccl import Cosmology
# import numpy as np
# from ../tk3d import Tk3D_SSC_Terasawa22

# # Creating a "Cosmology" object from the pyccl lib.
# # Passing to Tk3D_SSC_Terasawa22 func. with some arrays of k and a values
# # Checking the resulting object are:
# # of the correct type, shape and value. 
# # Done by the "assert" statement-> an error if condition not met.

# def test_Tk3D_SSC_Terasawa22():
#     cosmo = pyccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.7, sigma8=0.83,
#                       n_s=0.96, transfer_function='boltzmann_camb', matter_power_spectrum='camb')
    
#     #cosmo = pyccl.Cosmology(
#     #    Omega_c=0.25,
#     #    Omega_b=0.05,
#     #    h=0.7,
#     #    n_s=0.96,
#     #    sigma8=0.8,
#     #    matter_power_spectrum="linear"
#     #)

#     k_arr = np.logspace(-2, 3, 100)
#     a_arr = np.linspace(0.1, 1.0, 100)

#     tk = Tk3D_SSC_Terasawa22(
#         cosmo=cosmo,
#         lk_arr=np.log(k_arr),
#         a_arr=a_arr,
#         extrap_order_lok=1,
#         extrap_order_hik=1,
#         use_log=False,
#         deltah=0.02
#     )

#     #tk3d_ssc = tk3d.Tk3D_SSC_Terasawa22(cosmo=cosmo, deltah=0.02)
    
#     #tk3dssctets = tk3d_ssc.get_trispectrum(a_arr, k_arr)
#     #assert tk3dssctets == (len(a_arr), len(k_arr))

#     # Check the results is correct type
#     assert isinstance(tk, pyccl.tk3d.Tk3D)
    
#     # Check the shape/size of Tk3D object is correct
#     assert tk.lk_arr.size == k_arr.size
#     assert tk.a_arr.size == a_arr.size
#     assert tk.tk_arr.shape == (a_arr.size, k_arr.size)
#     # Calculate the power spectrum using Tk3D_SSC_Terasawa22 function

#     assert np.all(np.isfinite(tk))
    
#     # k = np.logspace(-2, 3, 100)
#     # z = 0.

#     # # Calculate the power spectrum using ccl's linear_matter_power function
#     # pk_ccl = pyccl.power.nonlin_matter_power(cosmo, k, a=1./(1+z))

#     # # Get the power spectrum from the Tk3D object
#     # pk_tk3d = tk3d_ssc.eval(k, 1./(1+z))

#     # # Check if the value are equal within some tolerance of error
#     # assert np.allclose(pk_ccl, pk_tk3d)

#     # # Plot the results for comparison
#     # plt.loglog(k, pk_tk3d, label='Tk3D_SSC_Terasawa22')
#     # plt.loglog(k, pk_ccl, label='ccl')
#     # plt.legend()
#     # plt.xlabel(r'$k\ [h\ \mathrm{Mpc}^{-1}]$')
#     # plt.ylabel(r'$P(k)\ [h^{-3}\ \mathrm{Mpc}^3]$')
#     # plt.show()
    
   
