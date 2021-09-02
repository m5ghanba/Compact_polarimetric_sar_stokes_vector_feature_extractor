# Compact_polarimetric_sar_stokes_vector_feature_extractor
Extractor of some compact polarimetric (CP) SAR features from the Stokes vector data 

Features used in the paper: Ghanbari, M., Clausi, D., A., Xu, L., and Jiang, M., (2019, Published). “Contextual Classification of Sea-Ice Types Using Compact Polarimetric SAR Data”. IEEE Transactions on Geoscience and Remote Sensing.

Deriving features from CP 2*2 coherence matrix (or equivalently Stokes vector)

Input: the elements of coherence matrix: c11, c12_real, c22, c12_imag
Output: The derived CP features:
1) scattering mechanism, alpha_s     2) circular polarization ratio, miu_c            3) conformity coefficient (ellipticity), u
4) correlation coefficient, rho      5) relative phase angle between RH and Rv, delta 6) degree of polarizatio, m
7) Shannon entropy, intensity, h_i   8) Shannon entropy, polarimetric component, h_p  9) degree of linear polar, m_l
10) linear polarization ratio, miu_l 11) orientation of the ellipse, psi              12) axial ratio of the ellipse, r
13 - 15) m-chi decomp, mchi_b, mchi_r, mchi_g              16 - 18) m-delta decomp, m_delta_r, m_delta_g, m_delta_b
19 - 20) RH and RV intensities, rh, rv                     21 - 24) Stokes parameters, s0, s1, s2, s3
