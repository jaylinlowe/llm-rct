import argparse
import pandas as pd
from openai import OpenAI
from transformers import set_seed
import logging
import time
import requests
from bs4 import BeautifulSoup
import re


def initialize_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)

def clean_text(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'\s+', ' ', text).strip()


def run_pipeline(args):

    logging.basicConfig(
    filename=f"open_access_log_{args.journal}_{args.question_type}.log",              
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s' )
    # Initialize OpenAI client
    client = initialize_openai_client(args.api_key)

    input_filename = f"open_access_journal_data/{args.journal}.csv"
    papers = pd.read_csv(input_filename)

    papers['id'] = range(1, len(papers) + 1)

    df1 = papers.copy()
    df2 = papers.copy()
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    df1['key'] = 1
    df2['key'] = 1

    df1['idx'] = df1.index
    df2['idx'] = df2.index
    pairs = pd.merge(df1, df2, on='key', suffixes=('.1', '.2'))

    if args.all_pairs == "False": 
        pairs = pairs[pairs['idx.1'] < pairs['idx.2']]
    elif args.all_pairs == "True":
        pairs = pairs[pairs['idx.1'] != pairs['idx.2']]
        pairs = pairs.sample(frac=1).reset_index(drop=True) # randomly shuffle order 
    else:
        raise ValueError("all_pairs must be True or False")

    pairs = pairs.drop(columns=['key', 'idx.1', 'idx.2']).reset_index(drop = True).reset_index(drop = True)

    pairs['abstract.1'] = pairs['abstract.1'].apply(clean_text)
    pairs['abstract.2'] = pairs['abstract.2'].apply(clean_text)   

    if args.journal == "genetics":
        journal = "Genetics"
    elif args.journal == "physio":
        journal = "Journal of Applied Physiology"
    elif args.journal == "faseb":
        journal = "The Federation of American Societies for Experimental Biology Journal"
    elif args.journal == "neuro":
        journal = "Journal of Neurophysiology"
    elif args.journal == "science":
        journal = "Science"

    
    if args.question_type == "basic":
        pairs['question'] =  "I have information on two papers published in " + journal + ". Using only the information I give you, which paper do you believe will have more citations? Paper 1 was published in " + pairs['Year.1'].astype(str) + " and was titled '" + pairs['Article Title.1'] + "'. Paper 1's abstract is: '" + pairs['abstract.1'] + "'. "
        pairs['question'] += " Paper 2 was published in " + pairs['Year.2'].astype(str) + " and was titled '" + pairs['Article Title.2'] + "'. Paper 2's abstract is: '" + pairs['abstract.2'] + "'. "

        if args.explanations == "False":
            pairs['question'] += " Please respond either 'Paper 1' or 'Paper 2' with no additional text."
        elif args.explanations == "True":
            pairs['question'] += " Please respond either 'Paper 1' or 'Paper 2' followed by a short one sentence explanation of your reasoning."

    elif args.question_type == "qualities":
        pairs['question'] =  "I have information on two papers published in " + journal + "Paper 1 was published in " + pairs['Year.1'].astype(str) + " and was titled '" + pairs['Article Title.1'] + "'. Paper 1's abstract is: '" + pairs['abstract.1'] + "'. "
        pairs['question'] += " Paper 2 was published in " + pairs['Year.2'].astype(str) + " and was titled '" + pairs['Article Title.2'] + "'. Paper 2's abstract is: '" + pairs['abstract.2'] + "'. "
        pairs['question'] += " I would like you to decide which paper best exhibits each one of these 11 qualities. The qualities are: topic novelty, topic popularity, title catchiness, generalizability, writing quality, impact of results, subfield popularity, technicality, meaningful contributions, journal fit for Science, and applicability. "
        pairs['question'] += " For each quality, please respond 'Paper 1' or 'Paper 2' depending on which one you think best exhibits the quality. Your answer should consist of 11 responses in order, each separated by commas."
        

        if args.explanations == "False":
            pairs['question'] += " You should not include any additional text beyond the 11 responses in order."
        elif args.explanations == "True":
            pairs['question'] += " After the 11 responses in order, please include a short one sentence explanation of your reasoning."


    elif args.question_type == "qualities_separate": 
        for quality in ['topic_novelty', 'topic_popularity', 'title_catchiness', 'generalizability', 'writing_quality', 'impact_of_results', 'subfield_popularity', 'technicality', 'meaningful_contributions' ,'journal_fit', 'applicability']:
            pairs[f'question_{quality}'] =  "I have information on two papers published in " + journal + "Paper 1 was published in " + pairs['Year.1'].astype(str) + " and was titled '" + pairs['Article Title.1'] + "'. Paper 1's abstract is: '" + pairs['abstract.1'] + "'. "
            pairs[f'question_{quality}'] += " Paper 2 was published in " + pairs['Year.2'].astype(str) + " and was titled '" + pairs['Article Title.2'] + "'. Paper 2's abstract is: '" + pairs['abstract.2'] + "'. "
            pairs[f'question_{quality}'] += f"I would like you to decide which paper best exhibits the quality {quality}"

            if args.explanations == "False":
                pairs[f'question_{quality}'] += " Please respond either 'Paper 1' or 'Paper 2' with no additional text."
            elif args.explanations == "True":
                pairs[f'question_{quality}'] += " Please respond either 'Paper 1' or 'Paper 2' followed by a short one sentence explanation of your reasoning."
        
    for i in range(0, len(pairs)):
    #for i in range(0, 5): 

        if (args.question_type == "basic") or (args.question_type == "qualities"):
            question = pairs.iloc[i]['question']
            max_output_tokens = 16000
            try:
                response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are an AI expert in academic scientific research."},
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

                if args.all_pairs == "False": 
                    filename = f"open_access_results/{args.journal}_{args.question_type}.csv"
                elif args.all_pairs == "True":
                    filename = f"open_access_results/{args.journal}_{args.question_type}_allpairs.csv"
                pairs.to_csv(filename, index = False)
        
        elif args.question_type == "qualities_separate":
            for quality in ['topic_novelty', 'topic_popularity', 'title_catchiness', 'generalizability', 'writing_quality', 'impact_of_results', 'subfield_popularity', 'technicality', 'meaningful_contributions' ,'journal_fit', 'applicability']:
                question = pairs.iloc[i][f'question_{quality}']
                max_output_tokens = 16000
                try:
                    response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": "You are an AI expert in academic scientific research."},
                        {"role": "user", "content": question},
                    ],
                    seed=args.seed,
                    max_tokens=max_output_tokens,
                )
                    pairs.at[i, f'{quality}'] = response.choices[0].message.content.strip()
                except Exception as e:
                    logging.error(f"Error occurred: {e}")
                    logging.exception(f"An error occurred for i = {i} and quality {quality}")

                if i % 200 == 0:
                    print(f"Done with index {i} and quality {quality}")

                    if args.all_pairs == "False": 
                        filename = f"open_access_results/{args.journal}_{args.question_type}.csv"
                    elif args.all_pairs == "True":
                        filename = f"open_access_results/{args.journal}_{args.question_type}_allpairs.csv"
                    pairs.to_csv(filename, index = False)
    
    pairs.to_csv(filename, index = False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="GPT model for aspect assignment (default: gpt-4o-mini)")
    parser.add_argument("--seed", type=int, default=93482)
    #parser.add_argument("--results_file", type = str, required = True)
    parser.add_argument("--explanations", type = str, required = True, default = "False")
    parser.add_argument("--question_type", type = str, default = "basic") # can be "basic" or "qualities" or "qualities_separate" where each quality is posed as a a separate question 
    parser.add_argument("--journal", type = str)
    parser.add_argument("--all_pairs", type = str, default = "False") # can be False (only i > j pairs) or True (all pairs, in a randomized order)
    args = parser.parse_args()

    run_pipeline(args)