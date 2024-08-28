# Readme.md

- update 2024/04/22
- update 2024/07/05
- Last update 2024/08/28

- info from GELATO : https://github.com/TheSkyentist/GELATO

- How to use gelato


      python runGELATO.py PARAMS.json --single spectrum.fits z
      python runGELATO.py PARAMS.json ObjectList.fits 


## single spectra

### Fit

     python  ../Convenience/runGELATO.py ExampleParameters.json  --single ../specgelato/v0/specgelato_SPEC3.fits  0.69

### Generate plot files

     python  ../Convenience/plotResults.py ExampleParameters.json --single ../specgelato/v0/specgelato_SPEC3.fits  0.69


## multi spectra

Input Spectra files are stored in **../spec_forgelato**, with different folders below


      ls ../spec_forgelato
      v0 v1 v2 v3 v4 

- first make a symbolic link such tht the relative path of spectra is correct

      ln -s ../spec_forgelato spec_forgelato

## Where these spectra has been generated
- run GELATO (spectra were generated from **Fors2pcigale/docs/notebooks/fors2_emission_lines/AccessFors2Spectra_calculatebackground.ipynb**, version3 has larger spectra errors than version4) 


## Input parameter files

To run gelato, one need the configuration file for gelato which is a json file.

    - ExampleParameters.json
    - ExampleParametersFitInNb.json
    - ExampleParametersFitInNb_recoveryv3.json
    - ExampleParametersFitInNb_v2.json
    - ExampleParametersFitInNb_v3.json
    - ExampleParametersFitInNb_v4.json

It is recommended to edit this file at least to specify the the directory (OutputFolder) for the fit output.

## Object list

These files include the spectrum file path, the spectrum name and the redshift.

    - object_filelist_v0.fits
    - object_filelist_v1.fits
    - object_filelist_v2.fits
    - object_filelist_v3.fits
    - object_filelist_v4.fits

### Rregnerate Object sublist

- **ExtractObjectListSubSamples.ipynb** : read old list and extract the last object and make a short list.

## Run Gelato Fit on multiple spectra

      python ../Convenience/runGELATO.py ExampleParameters.json object_filelist_v0.fits
      python ../Convenience/runGELATO.py ExampleParametersFitInNb_v3.json object_filelist_v3.fits
      python ../Convenience/runGELATO.py ExampleParametersFitInNb_v4.json object_filelist_v4.fits


- Note its better to fit spectra from the command-line compared inside a notebook. Probably the fits can fail in a notebook, probably because missing memory or uncleared memory.

## Calculate Equivalent width 

      python ../Convenience/ewResults.py ExampleParameters.json object_filelist_v0.fits
      python ../Convenience/ewResults.py ExampleParametersFitInNb_v3.json object_filelist_v3.fits
      python ../Convenience/ewResults.py ExampleParametersFitInNb_v4.json object_filelist_v4.fits

## Notebooks developped here

Different versions of extracting and calibrating the spectra including the errors.


- **SSPFromGelato.ipynb** : Show the set of Emiles continuum spectra used by Gelato (input model).

- **Example.ipynb** : original notebook provided by GELATO  
- **ProcessMultiSpectraInitial.ipynb** : early adaptation of GELATO notebook  

- **ViewFitResultOneSpectrum.ipynb** : View spectra plots fitted as in the pdf. Overwritten plot functions. Analyse outputs from GELATO fit results on a single spectrum. Do not run a fit. Plot histograms and spectra à la Gelato inside the notebook. Plotting functions are implemented in this notebook.
- 
- **ViewFitResultMultipleSpectra.ipynb** :   same as *ViewFitResultOneSpectrum.ipynb* but for many Spectra : Analyse outputs from GELATO fit results for multipe spectra. Do not run a fit. Plot histograms and spectra à la Gelato inside the notebook. Plotting functions are implemented in this notebook. Extract info and  calculate pulls and emission lines.
Note these info/emission-lines results are not saved.


- **ExampleFitInNb.ipynb** : Fit in notebooks selected spectrum. Use the ExampleParametersFitInNb.json which defines ``ResultsFitInNb/``
Redo the fit of a single spectrum. Extract the information from fits results and show the nice plots in the notebook. Save the emissionlines_table in csv file. The plot functions are implemented inside the notebook.


- **ExampleFitInNb_simple.ipynb** :  Similar to the *ExampleFitInNb.ipynb* notebook. However the plotting routine are imported from  *ExampleFitNb.py*

- **ExampleFitInNb_loop.ipynb**: Similar to *ExampleFitInNb_simple.ipynb* but process Fits in a loop to process the spectra. Notice many fits one after the other may fails to identify emission lines probably due to lack of memory ??? Better use the *command python ../Convenience/runGELATO.py ...

   - may do the fits of spectra (optionnal in cas it has been fitted already)
   - may show nice plots (activate them manually)
    - extract information about the status and info about the processing
   - save info about the emission-lines fitted.

- **DumpRunProcessingStatus.ipynb**:  Dump the file containing general info on the processing. It handle files generated in *ExampleFitInNb_loop.ipynb*:



To understand why some fit were good or bad these notebook were implemented. May 2024.

- **ExampleFitInNb_Review_loop.ipynb**: Review all fitted spectra or not fitted
- **ExampleFitInNb_ReviewGoods_loop.ipynb**: Review only good fitted spectra, not bad
- **ExampleFitInNb_ReviewBads_loop.ipynb**: Review list of bads

- **ExampleFitInNb_recoverfitfailure.ipynb**: recover bad fit by splitting spectrum in two parts


But this notebook allowed to fix the spectrum continuum model inside gelato (August 2024).

- **ExampleFitInNb_AnalyseBuildingAndFittingModel.ipynb**: debug fit of continuum.

