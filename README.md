# Verification of Multiline TRL Calibration

A technique for assessing the validity of multiline TRL calibration. The main goal of the method is to identify error in the reference impedance of the calibration. You can find in-depth details about the method in [1]. The mTRL algorithm used is based on [2], which you can access here: https://github.com/ZiadHatab/multiline-trl-calibration

## What’s in this repo

The code in this repo includes everything I used to generate the plots in [1]. The measurement are raw S-parameters from the VNA.

Basically, the code covers three aspects:

1. The main code, which generates the final results. The folder `csv` contain the reflection coefficient uncertainties, which are computed through linear uncertainty propagation (see the python script in the folder `csv`) . The partial derivatives I used to propagate the uncertainties are from the “Analytic Derivatives” capability that ANSYS HFSS offers. I included the HFSS project I used in case you want to test it yourself (it is just a simple microstrip line). The project was created and tested in ANSYS AEDT 2021. I didn’t try this, but I think it should also work in the student version: [https://www.ansys.com/academic/students/ansys-electronics-desktop-student](https://www.ansys.com/academic/students/ansys-electronics-desktop-student)
2. The code for the theoretical part of sensitivity analysis of mTRL calibration is in the folder `_sensitivity_analysis`.
3. Some equations that I presented in [1] were derived with the help of the symbolic math package `sympy`. The code is in the folder `_symbolic_math`.

## ****Code requirements****

I used a combination of couple of well-known packages:

```powershell
python -m pip install -U numpy matplotlib sympy scikit-rf pandas scipy
```

Also, you need to have the files `mTRL.py`, `MultiCal.py`, `TUGmTRL.py` in the same folder.

## ****References****

- [1] Z. Hatab, M. Gadringer, A. Alterkawi, and W. Bösch, "An Impedance Transition Method to Verify the Reference Impedance of Multiline TRL Calibration," e-print: [https://arxiv.org/abs/2209.09163](https://arxiv.org/abs/2209.09163)
- [2] Z. Hatab, M. Gadringer and W. Bösch, "Improving The Reliability of The Multiline TRL Calibration Algorithm," 2022 98th ARFTG Microwave Measurement Conference (ARFTG), 2022, pp. 1-5, doi: [10.1109/ARFTG52954.2022.9844064](https://ieeexplore.ieee.org/document/9844064)