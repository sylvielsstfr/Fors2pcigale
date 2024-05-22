# Readme.md

- Last update 2024/04/22
- Last update 2024/05/22

- info from GELATO : https://github.com/TheSkyentist/GELATO

- How to use gelato

 

     python runGELATO.py PARAMS.json --single spectrum.fits z


     python runGELATO.py PARAMS.json ObjectList.fits 


## single spectra
python  ../Convenience/runGELATO.py ExampleParameters.json  --single ../specgelato/v0/specgelato_SPEC3.fits  0.69
python  ../Convenience/plotResults.py ExampleParameters.json --single ../specgelato/v0/specgelato_SPEC3.fits  0.69


## multi spectra

- first make a symbolic link such tht the relative path of spectra is correct

ln -s ../spec_forgelato spec_forgelato

- run GELATO
python ../Convenience/runGELATO.py ExampleParameters.json object_filelist_v0.fits


## Notebooks developped here

- *Example.ipynb* : original notebook provided by GELATO  

- *ProcessMultiSpectraInitial.ipynb* : early adaptation of GELATO notebook  

- *ViewFitResultOneSpectrum.ipynb* : View spectra plots fitted as in the pdf. Overwritten plot functions

- *ViewFitResultMultipleSpectra.ipynb* :   same as *ViewFitResultOneSpectrum.ipynb* but for many Spectra


- *ExampleFitInNb.ipynb* : Fit in notebooks selected spectrum