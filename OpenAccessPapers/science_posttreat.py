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
    filename="open_access_log_science_posttreat.log",              
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s' )
    # Initialize OpenAI client
    client = initialize_openai_client(args.api_key)
    papers = pd.read_csv("open_access_journal_data/science.csv")
    papers['abstract'] = papers['abstract'].apply(clean_text)

    #papers['question'] = "I have a paper with the title: '" + papers['Article Title'] + "'. What academic journal would you recommend I try to publish this paper in? You should make use of any information you have about papers with similar titles and where they are published. Please give a single journal name, with no additional text."
    #papers['question'] = "I have a paper with the title: '" + papers['Article Title'] + "'. I am considering submitting it to Nature, Science, PNAS, or PLOS One. Of these, what academic journal would you recommend I try to publish this paper in? You should make use of any information you have about papers with similar titles and where they are published. Please give a single journal name, with no additional text."
    #papers['question'] = "I have a paper with the title: '" + papers['Article Title'] + "'. I am considering submitting it to Science, Nature, PNAS, or PLOS One. Of these, what academic journal would you recommend I try to publish this paper in? You should make use of any information you have about papers with similar titles and where they are published. Please give a single journal name, with no additional text."  
    #papers['question'] = "I have a paper with the title: '" + papers['Article Title'] + "'. I am considering submitting it to Nature, PNAS, PLOS One, or Science. Of these, what academic journal would you recommend I try to publish this paper in? You should make use of any information you have about papers with similar titles and where they are published. Please give a single journal name, with no additional text. You must choose one of the four journals listed." 
    # 
    #papers['question'] = "I have a paper with the title: '" + papers['Article Title'] + "' and abstract: '" + papers['abstract'] + "'. What academic journal would you recommend I try to publish this paper in? You should make use of any information you have about papers with similar titles or abstracts and where they are published. Please give a single journal name, with no additional text." 
    papers['question'] = "I have a paper with the title: '" + papers['Article Title'] + "' and abstract: '" + papers['abstract'] + "'. I am considering submitting it to Nature, Science, PNAS, or PLOS One. Of these, what academic journal would you recommend I try to publish this paper in? You should make use of any information you have about papers with similar titles or abstracts and where they are published. Please give a single journal name, with no additional text. You must choose one of the four journals listed." 

    for i in range(0, len(papers)):
    #for i in range(0, 5): 

        
            question = papers.iloc[i]['question']
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
                papers.at[i, 'response'] = response.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"Error occurred: {e}")
                logging.exception(f"An error occurred for i = {i}")

            if i % 50 == 0:
                print(f"Done with index {i}")
                papers.to_csv("open_access_results/science_posttreat_abstracts2.csv", index = False)
    
    papers.to_csv("open_access_results/science_posttreat_abstracts2.csv", index = False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="GPT model for aspect assignment (default: gpt-4o-mini)")
    parser.add_argument("--seed", type=int, default=3726)
    args = parser.parse_args()

    run_pipeline(args)