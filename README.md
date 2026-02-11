# llm-rct: Leveraging Large Language Models to Improve Precision in Randomized Controlled Trials
This repository includes supplemental code for "Leveraging Large Language Models to Improve Precision in Randomized Controlled Trials" by Jaylin Lowe, Adam Sales, and Johann Gagnon-Bartsch. There are four datasets, each with its own folder. The code is organized as follows:


### CTA
Includes processing data for the Cognitive Tutor Algebra (CTA) example. Data is private and cannot be included. Files are:

* `cta_basic.ipynb` - File obtaining basic summary statistics given in the paper. Requires "cta_student_level_final.csv", which is student level and cannot be shared. 
* `cta_pipeline.py` - Main file for obtaining LLM predictions. Requires "cta_student_level_final.csv". Outputs "cta_results/pair_results.csv". 
* `cta_analysis.ipynb` - Analysis file of CTA results, based on output files generated in cta_pipeline.py. Requires "cta_results/pair_results.csv". 

### Recidivism
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


### Tweeted Papers

* `tweets_pipeline.py` - Main file for obtaining LLM predictions. 
* `data`
    * `twitter_raw_data.xlsx` - Raw data file downloaded directly from the paper ["To tweet or not to tweet, that is the question: a randomized trial of twitter effects in medical education](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0223992)
    * `twitter_raw_data_with_abstracts.xlsx` - Data file with unprocessed data plus abstracts. Abstracts were obtained manually. 
    * `twitter_raw_data_with_extra_info.xlsx` - Data file with unprocessed original data, abstract, and additional information like author affiliation. This file can be generated from `tweets_pipeline.py` with `get_data = "True"`
* `tweet_results` 
    * `tweets_pairs_with_response.csv` - results file for paired papers where the LLM was asked to predict the outcome.
    * `tweets_pairs_multiple_covariates.csv` - results file for paired papers where the LLM was asked to extract multiple covariates. 
* `tweets_analysis.ipynb` - Analysis file 



### Open Access Papers
