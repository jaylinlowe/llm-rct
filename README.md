# llm-rct: Leveraging Large Language Models to Improve Precision in Randomized Controlled Trials
This repository includes supplemental code for "Leveraging Large Language Models to Improve Precision in Randomized Controlled Trials". There are three datasets, each with its own folder. The code is organized as follows:


### Case Study 1: Open Access Papers
Includes processing data for the third case study. Data is private and cannot be included. 

* `open_access_preprocess.ipynb` - Pulls abstracts and preprocesses raw data. Creates journal-specific CSV files with RCT covariates, outcome, and abstract. 
* `openaccess_pipeline.py` - Main file for obtaining LLM predictions. Requires corresponding journal data file.  
* `open_access_analysis.ipynb` - Analysis file of results, based on output files generated in open_access_pipeline.py. 
* `science_posttreat.py` - Additional file for small experiment checking for post-treatment likelihood on Science papers only.
* `science_posstreat_analysis.ipynb` - Analysis file of small experiment on Science papers. 


### Case Study 2: Recidivism
Includes preprocessing, basic summary statistics, and all analysis for the Sentencing of Defendants and Recidivism example. Data is public and can be obtained from the `cpt.paper` R package and is also included here. 

* `judges.csv` - Unprocessed data file downloaded directly from the `cpt.paper` package. 
* `recividism_basic.ipynb` - File for basic summary statistics given in the paper. 
* `recidivism_pipeline.py` - Main file for obtaining LLM predictions. Outputs "recividism_results/pair_results_sizeN_allpairs.csv" where N is a parameter determining how large the stratum are. 
* `recidivism_analysis.ipynb` - Analysis file of recividism results, based on "recividism_results/pair_results_sizeN_allpairs.csv" file. 


### Case Study 3: CTA
Includes processing data for the Cognitive Tutor Algebra (CTA) example. Data is private and cannot be included. Files are:

* `cta_basic.ipynb` - File obtaining basic summary statistics given in the paper. Requires "cta_student_level_final.csv", which is student level and cannot be shared. 
* `cta_pipeline.py` - Main file for obtaining LLM predictions. Requires "cta_student_level_final.csv". Outputs "cta_results/pair_results_allpairs.csv". 
* `cta_analysis.ipynb` - Analysis file of CTA results, based on output files generated in cta_pipeline.py. Requires "cta_results/pair_results_allpairs.csv". 
* `cta_paper_results.Rmd`- R file of CTA results, based on output files generated in cta_analysis.ipynb. 

