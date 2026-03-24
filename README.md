# llm-rct: Leveraging Large Language Models to Improve Precision in Randomized Controlled Trials
This repository includes supplemental code for "Leveraging Large Language Models to Improve Precision in Randomized Controlled Trials" by Jaylin Lowe, Adam Sales, and Johann Gagnon-Bartsch. There are three datasets, each with its own folder. The code is organized as follows:

### Case Study 1: Recidivism
Includes preprocessing, basic summary statistics, and all analysis for the Sentencing of Defendants and Recidivism example. Data is public and can be obtained from the `cpt.paper` R package and is also included here. 

* `judges.csv` - Unprocessed data file downloaded directly from the `cpt.paper` package. 
* `recividism_basic.ipynb` - File for basic summary statistics given in the paper. 
* `recividism_pipeline.py` - Main file for obtaining LLM predictions. Outputs "recividism_results/pair_results_sizeN.csv" where N is a parameter determining how large the stratum are. 
* `recividism_analysis.ipynb` - Analysis file of recividism results, based on "recividism_results/pair_results_sizeN.csv" file. 
* `preprocessed_data`
    * `judges_pair_preprocessed_unpaired_size100.csv` - preprocessed file prior to pairing
    * `judges_pair_preprocessed_size100.csv` - preprocessed file after pairing 
* `recividism_results`
    * `pair_results_size100.csv` - results file used in paper, when dataset is paired into stratum with approximately 100 observations per stratum 



### Case Study 2: CTA
Includes processing data for the Cognitive Tutor Algebra (CTA) example. Data is private and cannot be included. Files are:

* `cta_basic.ipynb` - File obtaining basic summary statistics given in the paper. Requires "cta_student_level_final.csv", which is student level and cannot be shared. 
* `cta_pipeline.py` - Main file for obtaining LLM predictions. Requires "cta_student_level_final.csv". Outputs "cta_results/pair_results.csv". 
* `cta_analysis.ipynb` - Analysis file of CTA results, based on output files generated in cta_pipeline.py. Requires "cta_results/pair_results.csv". 


### Case Study 3: Open Access Papers
Includes processing data for the third case study. Data is private and cannot be included. 

* `open_access_preprocess.ipynb` - Pulls abstracts and preprocesses raw data. Creates journal-specific CSV files with RCT covariates, outcome, and abstract. 
* `open_access_pipeline.py` - Main file for obtaining LLM predictions. Requires corresponding journal data file.  
* `open_access_analysis.ipynb` - Analysis file of results, based on output files generated in open_access_pipeline.py
* `science_posttreat.py` - Additional file for small experiment checking for post-treatment likelihood on Science papers only.
* `science_posstreat_analysis.ipynb` - Analysis file of small experiment on Science papers. 


