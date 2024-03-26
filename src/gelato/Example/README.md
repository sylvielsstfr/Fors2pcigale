# Readme.md



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


