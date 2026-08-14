# This is the full code pipeline for the CTA data
# it's based on cta_student_level_final.csv, which is created in the R script create_student_level.R saved locally and rerun on 1/26 so it's up to date
# you can rerun the preprocessing steps to create that file or if already exists, skip that and run the chatGPT part 
# it pairs students and asks chatGPT to predict which student will score higher 
# this version runs all pairs - i.e. (i,j) and (j,i) are both run 


import argparse
import pandas as pd
from openai import OpenAI
from transformers import set_seed
import logging
import time
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


def initialize_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def get_oob_preds(df):
    
    num_cols = df.select_dtypes(include="number").columns

    #mean imputation and new column creation for missing values
    for col in num_cols:
        if df[col].isna().any():
            # Missingness indicator
            df[f"{col}_mis"] = df[col].isna().astype(int)

            # Mean imputation
            df[col] = df[col].fillna(df[col].mean())

    df = df.copy()

    y = df['y_yirt']
    X = df.drop(columns = ['y_yirt'])


    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns

    preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

    model = RandomForestRegressor(
        oob_score = True,
        bootstrap = True,
        random_state=2772,
        n_jobs=-1
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocess),
            ("rf", model)
            ]
    )

    pipe.fit(X, y)
    rf = pipe.named_steps["rf"]

    oob_preds=  rf.oob_prediction_
    return oob_preds



def preprocess(args): 
    cta = pd.read_csv("cta_student_level_final.csv")
    cta = cta[['state', 'grdlvl', 'race', 'sex', 'spec_speced', 'spec_gifted', 'spec_esl', 'frl', 'y_yirt' ,'xirt']]
    cta['oob_preds'] = get_oob_preds(cta.copy())
    print("done with random forest")

    cta['id'] = range(1, len(cta) + 1)

    # get group based on oob_preds
    cta["_orig_order"] = cta.index
    cta = (
        cta
        .sort_values("oob_preds")
        .reset_index(drop=True)
    )

    cta["pair_group"] = cta.index // 10
    cta["pair_group_size"] = cta.groupby("pair_group")["pair_group"].transform("size")

    cta = cta.sort_values("_orig_order").drop(columns="_orig_order")

    # reformat covariates 
    cta['sex'] = cta["sex"].map({"M": "They are male, ", "F": "They are female, "}).fillna("We do not know their gender, ")
    cta['race'] = cta['race'].map({"WHITE NON-HISPANIC": "White non-Hispanic, ", 
                                "BLACK NON-HISPANIC": "Black non-Hispanic, " , 
                                "ASIAN / PACIFIC ISLANDER": "Asian or Pacific Islander, " , 
                                "OTHER RACE / MULTI-RACIAL": "multiracial or belong to a race category other than White, Black, Asian, Hispanic, or American Indian, " ,
                                "AMERICAN INDIAN / ALASKAN NATIVE": "American Indian or Alaskan Native, " , 
                                "HISPANIC": "are Hispanic, "}).fillna("their race is unknown, ")
    cta['frl'] = cta["frl"].map({"1.0": "and recieve free or reduced lunch. ", "0.0": "and do not recieve free or reduced lunch. "}).fillna("and whether they recieve free or reduced lunch is unknown. ")

    cta['spec_speced'] = cta["spec_speced"].astype(str).map({"1.0": "They are in special education, ", "0.0": "They are not in special education, "}).fillna("It is unknown if they are in special education, ")
    cta['spec_gifted'] = cta["spec_gifted"].astype(str).map({"1.0": "are in gifted education, ", "0.0": "are not in gifted education, "}).fillna("it is unknown if they are in gifted education, ")
    cta['spec_esl'] = cta["spec_esl"].astype(str).map({"1.0": "and are in ESL education. ", "0.0": "and are not in ESL education. "}).fillna("and it is unknown if they are in ESL education. ")

    cta['grdlvl'] = np.where(cta['grdlvl'] == "M", "middle school student", "high school student")
    state_map = {
        "TX": "Texas",
        "KY": "Kentucky",
        "LA": "Louisiana",
        "MI": "Michigan",
        "CT": "Connecticut",
        "AL": "Alabama",
        "NJ": "New Jersey"
    }

    cta['xirt'] = cta["xirt"].fillna("unknown").astype(str)

    cta["state"] = cta["state"].map(state_map)
    print("Done with relabeling and leveling variables")

    cta.to_csv("cta_preprocessed_unpaired.csv")
    

    # randomly reorder, and then merge only on group 
    df1 = cta.copy().sample(frac = 1, random_state = 9).reset_index(drop = True)
    df2 = cta.copy().sample(frac = 1, random_state = 9).reset_index(drop = True)
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    df1['idx'] = df1.index
    df2['idx'] = df2.index
    pairs = pd.merge(df1, df2, on='pair_group', suffixes=('.x', '.y'))

    #pairs = pairs[pairs['idx.x'] < pairs['idx.y']]
    pairs = pairs[pairs['idx.x'] != pairs['idx.y']]
    pairs = pairs.drop(columns=['idx.x', 'idx.y']).reset_index(drop = True)
    print("Done with pairing")




    pairs['question'] =  "I am going to give you information about two students. Both just took an algebra class at their school. Using only the information I give you, which student do you think will score higher on an algebra proficiency test administered at the end of the class? "
    pairs['question'] += "Student 1 is a " + pairs['grdlvl.x'] +  " in " + pairs['state.x'] + ". "
    pairs['question'] += pairs['sex.x'] + pairs['race.x'] + pairs['frl.x']
    pairs['question'] += pairs['spec_speced.x'] + pairs['spec_gifted.x'] + pairs['spec_esl.x']
    pairs['question'] += "Student 1's standardized score on an algebra readiness exam administred before the course was " + pairs['xirt.x'] + ". "


    pairs['question'] += "Student 2 is a " + pairs['grdlvl.y'] +  " in " + pairs['state.y'] + ". "
    pairs['question'] += pairs['sex.y'] + pairs['race.y'] + pairs['frl.y']
    pairs['question'] += pairs['spec_speced.y'] + pairs['spec_gifted.y'] + pairs['spec_esl.y']
    pairs['question'] += "Student 2's standardized score on an algebra readiness exam administred before the course was " + pairs['xirt.y'] + ". "
    
    pairs['question'] += " Please respond either 'Student 1' or 'Student 2' with no additional text."

    pairs.to_csv("cta_pairs_preprocessed_allpairs.csv", index = False)


def run_pipeline(args):

    logging.basicConfig(
    filename=f"cta_log.log",              
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s' )
    # Initialize OpenAI client
    client = initialize_openai_client(args.api_key)

    if args.run_preprocess == "False":
        pairs = pd.read_csv("cta_pairs_preprocessed_allpairs.csv")
    else: 
        preprocess(args)
        pairs = pd.read_csv("cta_pairs_preprocessed_allpairs.csv")
        
    for i in range(0, len(pairs)):
    #for i in range(0, 5): 

        question = pairs.iloc[i]['question']
        max_output_tokens = 16000
        try:
            response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are a knowledgable expert in education."},
                {"role": "user", "content": question},
            ],
            seed=args.seed,
            max_tokens=max_output_tokens,
        )
            pairs.at[i, 'response'] = response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            logging.exception(f"An error occurred for i = {i}")

        if i % 200 == 0:
            print(f"Done with index {i}")
            pairs.to_csv("cta_results/pair_results_allpairs.csv", index = False)
    
    pairs.to_csv("cta_results/pair_results_allpairs.csv", index = False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="GPT model for aspect assignment (default: gpt-4o-mini)")
    parser.add_argument("--seed", type=int, default=93482)
    #parser.add_argument("--results_file", type = str, required = True)
    parser.add_argument("--run_preprocess", type = str, default = "False")
    args = parser.parse_args()

    run_pipeline(args)