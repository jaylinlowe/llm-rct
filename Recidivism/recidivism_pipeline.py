# This is the full code pipeline for the Green and Winik data
# it's based on judges.csv, which is the unprocesed file downloaded from the R package cpt.paper (cite Johann's repo and paper)
# we will stratify on an OOB pred and run unordered pairs only, but in a random order 
# defaults to groups of size 100 (10 groups approximately) but that can be changed 


import argparse
import pandas as pd
from openai import OpenAI
from transformers import set_seed
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression


def initialize_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def get_oob_preds(df):
    
    y = df['laterarr']
    X = df.drop(columns = ['laterarr'])

    rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,       
    bootstrap=True,        
    random_state=8471,
    n_jobs=-1              
)

    rf.fit(X, y)
    oob_probs = rf.oob_decision_function_[:, 0] # gives OOB predicted probabilities (since logistic)
    return oob_probs

def preprocess(args): 
    judges = pd.read_csv("judges.csv")
    judges['id'] = range(1, len(judges) + 1)
    judges = judges[['id', 'age', 'gender', 'nonblack', 'marijuana', 'cocaine', 'crack', 'heroin', 'pcp', 'otherdrug', 'nondrug', 'priorarr', 'priorfelarr', 'priordrugarr', 'priorfeldrugarr', 'priorcon', 'priorfelcon', 'priordrugcon', 'priorfeldrugcon', 'pwid', 'dist', 'laterarr']]
    judges['gender'] = np.where(judges['gender'] == 'M', 0, 1) # Ws and Fs coerced to female
    judges['oob_preds'] = get_oob_preds(judges.copy())
    judges['gender'] = np.where(judges['gender'] == 0, "male", "female") # switch back to words for question below 
    judges['nonblack'] = np.where(judges['nonblack'] == 0, "not Black", "Black")

    drug_cols = ['marijuana', 'cocaine', 'crack', 'heroin', 'pcp']

    judges['selected_drugs'] = judges[drug_cols].apply(
        lambda row: ' and '.join(row.index[row == 1]), axis=1
    )


    judges['arrest_conviction'] = ''
    judges['pwid_dist'] = ''

    # give highest level of conviction + any arrests that they weren't convicted of 
    for i in range(len(judges)):
        person = judges.iloc[i]
        arrest_conviction = ''
        pwid_dist = ''

        # PEOPLE WITH NO ARRESTS OR CONVICTIONS
        if person['priorarr'] == 0:
            arrest_conviction = "They have no prior arrests."

        # FELONY DRUG CONVICTIONS 
        elif person['priorfeldrugcon'] == 1: 
            arrest_conviction = "In the past, they were arrested and convicted on a felony drug charge."
            # this person will have a 1 for all arrests, so we don't have to check anything here 

        # FELONY CONVICTIONS AND DRUG CONVICTIONS
        elif (person['priorfelcon'] == 1) and (person['priordrugcon'] == 1): 
            arrest_conviction = "In the past, they were arrested and convicted on two separate charges, one of which was a felony and one of which was drug related."
            #this person may have had a felony drug charge that they weren't convicted for 
            if person['priorfeldrugarr'] == 1:
                arrest_conviction += "In addition, they have been arrested for a felony drug charge in the past, but they weren't convicted of it. "

        # FELONY CONVICTIONS
        elif person['priorfelcon'] == 1:
            arrest_conviction = "In the past, they were arrested and convicted on a felony charge."

            # This person could have a drug charge that they weren't convicted for, or a felony drug charge
            if person['priorfeldrugarr'] == 1:
                arrest_conviction += "In addition, they have been arrested for a felony drug charge in the past, but they weren't convicted of it."
            elif person['priordrugarr'] == 1:
                arrest_conviction += "In additon, they have been arrested for a non-felony drug charge in the past, but they weren't convicted of it."

        # DRUG CONVICTIONS 
        elif person['priordrugcon'] == 1:
            arrest_conviction = "In the past, they were arrested and convicted on a non-felony drug related charge."

            # this person could have a felony charge they weren't convicted for, or a felony drug charge 
            if person['priorfeldrugarr'] == 1:
                arrest_conviction += "In addition, they have been arrested for a felony drug charge in the past, but they weren't convicted of it."
            elif person['priorfelarr'] == 1:
                arrest_conviction += "In additon, they have been arrested for a felony charge in the past, but they weren't convicted of it."

        # OTHER CONVICTIONS 
        elif person['priorcon'] == 1:
            arrest_conviction = "In the past, they were arrested and convicted of a crime, but it wasn't drug related or felony."

            #this person could have a felony charge, a drug charge, or a felony drug charge they weren't convicted for 
            if person['priorfeldrugarr'] == 1:
                arrest_conviction += "In addition, they have been arrested for a felony drug charge in the past, but they weren't convicted of it."
            elif (person['priorfelarr'] == 1) and (person['priorfeldrugarr'] == 1):
                arrest_conviction += "In additon, they have been arrested at least two other charges in the past, one a felony and one drug related, but they weren't convicted of either."
            elif person['priorfelarr'] == 1:
                arrest_conviction += "In additon, they have been arrested for a felony charge in the past, but they weren't convicted of it."
            elif person['priordrugarr'] == 1:
                arrest_conviction += "In additon, they have been arrested for a non-felony drug charge in the past, but they weren't convicted of it."

        # OTHER ARRESTS WITH NO CONVICTIONS 
        elif person['priorfeldrugarr'] == 1:
            arrest_conviction = "In the past, they were arrested on a felony drug charge, but they weren't convicted of it."

        elif (person['priorfelarr'] == 1) and (person['priordrugarr'] == 1): 
            arrest_conviction = "In the past, they were arrested on two separate charges: one felony charge and one drug-related charge, but they weren't convicted of either."

        elif person['priorfelarr'] == 1:
            arrest_conviction = "In the past, they were arrested on a felony charge, but they weren't convicted of it."

        elif person['priordrugarr'] == 1:
            arrest_conviction = "In the past, they were arrested on a non-felony drug related charge, but they weren't convicted of it."

        elif person['priorarr'] == 1:
            arrest_conviction = "In the past, they were arrested, but it wasn't a drug related or felony arrest, and they weren't convicted."
        

        if (person['pwid'] == 1) & (person['dist'] == 1):
            pwid_dist += "charged with possession and intent to distribute " 
        elif (person['pwid'] == 1) & (person['dist'] == 0):
            pwid_dist += "charged with possession and intent to distribute "
        elif (person['pwid'] == 0) & (person['dist'] == 1):
            pwid_dist += "charged with distribution of "

        if person['otherdrug'] == 0:
            pwid_dist += person['selected_drugs'] 
        elif (person['otherdrug'] == 1) & (person['selected_drugs'] == ''):
            pwid_dist += "a drug other than marijuana, cocaine, crack, heroin, or PCP. This may include prescription drugs or ectasy"
        elif (person['otherdrug'] == 1) & (person['selected_drugs'] != ''):
            pwid_dist += person['selected_drugs'] + "plus at least one other drug not in one of these categories"

        if person['nondrug'] == 1:
            pwid_dist += ". They also were charged with at least one non-drug related charge"


        judges.at[i, 'arrest_conviction'] =  arrest_conviction
        judges.at[i, 'pwid_dist'] = pwid_dist

    judges["_orig_order"] = judges.index
    judges = (
        judges
        .sort_values("oob_preds")
        .reset_index(drop=True)
    )

    judges["pair_group"] = judges.index // args.group_size # about 10 groups 
    if args.group_size == 100: 
        judges['pair_group'] = np.where(judges['pair_group'] == 10, 9, judges['pair_group']) #collapse group 10 into 9 since its so small
    judges["pair_group_size"] = judges.groupby("pair_group")["pair_group"].transform("size")
    
    judges.to_csv(f"preprocessed_data/judges_pairs_preprocessed_unpaired_size{args.group_size}_allpairs.csv")

    # randomly reorder, and then merge only on group 
    df1 = judges.copy().sample(frac = 1, random_state = 48741).reset_index(drop = True)
    df2 = judges.copy().sample(frac = 1, random_state = 482).reset_index(drop = True)
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)


    pairs = pd.merge(df1, df2, on='pair_group', suffixes=('.x', '.y'))
    pairs = pairs[pairs['id.x'] != pairs['id.y']]
    print("Done with pairing")


    # QUESTION FORMATTING 
    pairs = pairs.reset_index(drop = True)

    basic1 = "I have information on two specific defendants who were just arrested in Washington, DC on drug related charges. "

    basic2 = " Using your knowledge of criminal records and likelihood of reoffending, which of these two people is more likely to be arrested again, on any criminal charge, in the next four years? Please respond either 'Person 1' or 'Person 2' with no additional text."


    for i in range(len(pairs)):

        pair = pairs.iloc[i]
        question = ''
        
        question = basic1 + f'''Person 1 is {pair['gender.x']}, {pair['age.x']} years old, and was just arrested and {pair['pwid_dist.x']}.  {pair['arrest_conviction.x']} Person 2 is {pair['gender.y']}, {pair['age.y']} years old, and {pair['pwid_dist.y']}. {pair['arrest_conviction.y']}''' + basic2 

        pairs.at[i, 'question'] =  question 

    pairs.to_csv(f"preprocessed_data/judges_pairs_preprocessed_size{args.group_size}_allpairs.csv", index = False)


def run_pipeline(args):

    logging.basicConfig(
    filename=f"recidivism_results/recidivism_log.log",              
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s' )
    # Initialize OpenAI client
    client = initialize_openai_client(args.api_key)

    if args.run_preprocess == "True":
        preprocess(args)
   
    pairs = pd.read_csv(f"preprocessed_data/judges_pairs_preprocessed_size{args.group_size}_allpairs.csv")
        
    for i in range(0, len(pairs)):
    #for i in range(0, 5): 

        question = pairs.iloc[i]['question']
        max_output_tokens = 16000
        try:
            response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are an expert in the US justice system and understand the underlying patterns behind who commits crimes."},
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
            filename = f"recidivism_results/pair_results_size{args.group_size}_allpairs.csv"
    
    pairs.to_csv(filename, index = False)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="GPT model for aspect assignment (default: gpt-4o-mini)")
    parser.add_argument("--seed", type=int, default=93482)
    #parser.add_argument("--results_file", type = str, required = True)
    parser.add_argument("--run_preprocess", type = str, default = "False")
    parser.add_argument("--group_size", default = 100)
    args = parser.parse_args()

    run_pipeline(args)