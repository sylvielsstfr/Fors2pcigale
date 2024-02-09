# README.md

Run of pcigal SSP fit and Analysis of pcigale fits output on ForsData

- author Sylvie Dagoret-Campagne
- creation date February 2024
- update 2024-02-08

From document ands_on_project_CIGALE.pdf and article at https://www.aanda.org/articles/aa/full_html/2019/02/aa34156-18/aa34156-18.html

## How to run pcigale



### 1) run pcigale ini

This generate a first part of the pcigale configuration file to fill next.

### 2) fill the pcigale.ini file

Need to provide input files.
Here I provide two fits inputs Fors2Photom_cigaleinput_WithNan.fits and
Fors2Photom_cigaleinput_NONan.fits  

Complete the pcigale.ini file as follow:

- data_file = Fors2Photom_cigaleinput_NONan.fits
- sed_modules = sfhdelayed,bc03,nebular,dustatt_modified_starburst,dale2014,fritz2006,radio,restframe_parameters,redshifting
- analysis_method = pdf_analysis
- cores = 12

- metallicity = 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05 
- logU = -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6,-2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0


### 3) pcigale genconf

Add more info automatically in pcigale.ini

### 4) fill again pcigale.ini
- save_best_sed = True

### 5) pcigale check

- Check the format of pcigale.ini is correct

### 6) pcigale run

- run the full simulation

### 7) pcigale-plots sed
- let add pcigale-plots add SED plots for each spectrum

## Analyse of cigale results

Notebook SDC has written to analyse output resuts

### initial notebook from pcigale school
- cigale_manual_fitting.ipynb  

### notebook adapted for Fors2
- cigale_adjust_fitresults.ipynb    

### tool to view output of cigale in fits file
- DumpPcigaleFits.ipynb               

### view plot from single spectrum pcigale-fit
- cigale_view_fitresults.ipynb

### loop to view all fit results
cigale_view_fitresults-loop.ipynb    

### loop to save all fit results in a file
cigale_save_fitresults-loop.ipynb     
                                            
SaveFittedSSP.ipynb                 





