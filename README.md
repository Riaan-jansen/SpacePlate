README

15/08/25

NEW

SpacePlate/. contains all the code.

CODE:
 - modal_match.py: all the modal matching code with the experimental geometry.

 - data_collecting.py: for running the experiment.

 - data_processing.py: for analysing experimental data.

 - comparison_graphs.py: analysis of COMSOL files and comparison between all
    three approaches.

 - other code: not too important.

DATA:
 - scan_data/.
  - sp means space plate
  - ns means no space plate
  - number after sp/ns indicates distance from speaker to mic
  (so 'scan_ns55.npy' is 55 cm separation between speaker and mic, no plate)
  - *_x3 means measurements three times per unit cell (x_step = 6.7/3)
 
 - comsol_data/.
  - oldgeom refers to the first measured geometry of the space plate
    - radius = 1.25, pipe depth = 1.5, plate gap = 18.9 (mm)
  - both oldgeom are parametric sweep studies theta_step = 1
  - 'sp_oldgeom_TA.csv' lacks proper periodic conditions in the gap
  - 'sp_final_PA0.csv' is updated geometry, only theta = 0
  - 'sp_finalgeom_TA_t5.csv' is updated, periodic conditions on the gap
      but peaks are still shifted higher in frequency than expected.

 - pictures
  - now separated out into categories
    - compression: compression factor results
    - dispersion_relations: --
    - thesis_comparison: --
    - transmission: frequency spectra


08/07/25

CONTENTS
- SpacePlate
  - the code:
      - modal_match.py: implementing modal matching to produce plots of modes allowed single and double fishnet case
                            replicating Murray thesis work Figures 5.3-5.6.
      - mm_dispersion.py: modal matching, kx against k0 with Transmission coefficients plotted as colour. Figure 6.2.
      - sim_eqn_solver.py: to solve for coefficients for T
      - trig_solver.f90: to solve equation 5.13 in Murray thesis (impedence matching)
   
- papers
  - papers used

- pictures
    - picture outputs from code or otherwise
