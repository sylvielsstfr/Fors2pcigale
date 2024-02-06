# README.md

From document ands_on_project_CIGALE.pdf and article at https://www.aanda.org/articles/aa/full_html/2019/02/aa34156-18/aa34156-18.html

## 1) run pcigale ini

## 2) fill the pcigale.ini file

- data_file = Fors2Photom_cigaleinput_NONan.txt
- sed_modules = sfhdelayed,bc03,nebular,dustatt_modified_starburst,dale2014,fritz2006,radio,restframe_parameters,redshifting
- analysis_method = pdf_analysis
- cores = 12

- metallicity = 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05 
- logU = -4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6,-2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0



## 3) pcigale genconf

## 4) fill again pcigale.ini
- save_best_sed = True

## 5) pcigale check


## 6) pcigale run

## 7) pcigale-plots sed
