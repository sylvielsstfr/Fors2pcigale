# README.md

From document ands_on_project_CIGALE.pdf and article at https://www.aanda.org/articles/aa/full_html/2019/02/aa34156-18/aa34156-18.html

## 1) run pcigale ini

## 2) fill the pcigale.ini file

- data_file = Fors2Photom_cigaleinput_NONan.txt
- sed_modules = sfhdelayed,bc03,nebular,dustatt_modified_starburst,dale2014,restframe_parameters,redshifting
- analysis_method = pdf_analysis
- cores = 4

## 3) pcigale genconf

## 4) fill again pcigale.ini
- save_best_sed = True

## 5) pcigale check


## 6) pcigale run:

## 7) pcigale-plots sed
