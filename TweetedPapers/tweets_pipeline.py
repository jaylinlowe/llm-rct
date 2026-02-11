import argparse
import pandas as pd
from openai import OpenAI
from transformers import set_seed
import logging
import time
import requests
from bs4 import BeautifulSoup

def doi_to_pmid(doi: str):
    """
    Convert DOI to PubMed ID using NCBI E-utilities.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": f"{doi}[DOI]",
        "retmode": "xml"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "xml")
    ids = soup.find_all("Id")
    return ids[0].text if ids else None


def fetch_pubmed_metadata_and_abstract(pmid: str):
    """
    Given a PMID, retrieve abstract + metadata from PubMed.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "xml")

    # Title
    title_tag = soup.find("ArticleTitle")
    title = title_tag.get_text(" ", strip=True) if title_tag else None

    # Abstract
    abstract_tag = soup.find("Abstract")
    if abstract_tag:
        paragraphs = abstract_tag.find_all(["AbstractText"])
        abstract = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    else:
        abstract = None

    # Journal
    journal_tag = soup.find("Title")
    journal = journal_tag.get_text(" ", strip=True) if journal_tag else None

    # Year
    year_tag = soup.find("PubDate")
    year = None
    if year_tag:
        yr = year_tag.find("Year")
        if yr:
            year = yr.get_text(strip=True)

    # Authors
    authors = []
    for auth in soup.find_all("Author"):
        last = auth.find("LastName")
        fore = auth.find("ForeName")
        if last and fore:
            authors.append(f"{fore.get_text(strip=True)} {last.get_text(strip=True)}")

    return {
        "pmid": pmid,
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year,
        "abstract": abstract,
    }
def get_pubmed_authors_and_affiliations(pmid: str):
    """
    Fetch authors and their affiliations for a given PMID.
    """
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    
    soup = BeautifulSoup(r.text, "xml")
    authors = []

    for auth in soup.find_all("Author"):
        last = auth.find("LastName")
        fore = auth.find("ForeName")

        name = None
        if last and fore:
            name = f"{fore.get_text(strip=True)} {last.get_text(strip=True)}"
        elif last:
            name = last.get_text(strip=True)

        # Collect all affiliations for this author
        affs = [
            aff.get_text(" ", strip=True)
            for aff in auth.find_all("Affiliation")
        ]

        authors.append({
            "name": name,
            "affiliations": affs
        })

    return authors



def initialize_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def run_pipeline(args):

    logging.basicConfig(
    filename='openai_tweets.log',              
    level=logging.INFO,              
    format='%(asctime)s - %(levelname)s - %(message)s' )
    # Initialize OpenAI client
    client = initialize_openai_client(args.api_key)

    tweets = pd.read_csv("data/twitter_raw_data_with_abstracts.csv")

    if args.get_data == "True": 
        print("Getting extra paper metadata...")
        for i in range(0, len(tweets)): 
            pmid = doi_to_pmid(tweets['DOI (v31)'].iloc[i])
            data = fetch_pubmed_metadata_and_abstract(pmid)
            authors = get_pubmed_authors_and_affiliations(pmid)

            tweets.at[i, 'affiliations'] = authors[0]['affiliations'][0]
            names = data['authors']
            if not names:
                tweets.at[i, "authors"] = "The authors are unknown"
            elif len(names) == 1:
                tweets.at[i, "authors"] = "The author is " + names[0]
            elif len(names) == 2:
                tweets.at[i, "authors"] = f"The authors are {names[0]} and {names[1]}"
            else:
                tweets.at[i, "authors"] = f"The authors are {', '.join(names[:-1])}, and {names[-1]}"

            tweets.at[i, "title"] = data['title']
            tweets.at[i, "pmid_abstract"] = data['abstract']

            time.sleep(2)
        tweets.to_csv("data/twitter_raw_with_extra_info.csv")
        tweets_cleaned = tweets[['DOI (v31)', 'Abstract', 'authors', 'title', 'affiliations']]

    else: 
        tweets_cleaned = pd.read_csv("data/twitter_raw_data_with_extra_info.csv")

    if args.use_pairs == "True":
        df1 = tweets_cleaned.copy()
        df2 = tweets_cleaned.copy()
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        df1['key'] = 1
        df2['key'] = 1

        df1['idx'] = df1.index
        df2['idx'] = df2.index
        pairs = pd.merge(df1, df2, on='key', suffixes=('.1', '.2'))

        pairs = pairs[pairs['idx.1'] < pairs['idx.2']]

        pairs = pairs.drop(columns=['key', 'idx.1', 'idx.2']).reset_index(drop = True)

        if args.question_type == "basic": 

            if args.info_type == "all":
                if args.explanations == "False":
                    pairs['question'] = "I have information on two papers both published in Academic Medicine in 2015. Paper 1's title is: " + pairs['title.1'] + ". " + pairs['authors.1'] + ". " + pairs['affiliations.1'] + ". Paper 1's abstract is: '" +  pairs['Abstract.1'] +  "'. +  Paper 2's title is: " + pairs['title.2'] + ". " + pairs['authors.2'] + ". " + pairs['affiliations.2'] + ". Paper 2's abstract is: '" +  pairs['Abstract.2'] +  "'. Which paper do you believe will have more views? Please respond either 'Paper 1' or 'Paper 2' with no additional text."
                else: 
                    pairs['question'] = "I have information on two papers both published in Academic Medicine in 2015. Paper 1's title is: " + pairs['title.1'] + ". " + pairs['authors.1'] + ". " + pairs['affiliations.1'] + ". Paper 1's abstract is: '" +  pairs['Abstract.1'] +  "'. +  Paper 2's title is: " + pairs['title.2'] + ". " + pairs['authors.2'] + ". " + pairs['affiliations.2'] + ". Paper 2's abstract is: '" +  pairs['Abstract.2'] +  "'. Which paper do you believe will have more views? Please respond either 'Paper 1.' or 'Paper 2.' followed by a short one sentence explanation."

            elif args.info_type == "affiliations":
                if args.explanations == "False":
                    pairs['question'] = "I have information on two papers both published in Academic Medicine in 2015. For paper 1,  "  + pairs['authors.1'] + ". " + pairs['affiliations.1'] + ". +  For Paper 2,   " + pairs['authors.2'] + ". " + pairs['affiliations.2'] +  "'. Which paper do you believe will have more views? Please respond either 'Paper 1' or 'Paper 2' with no additional text."
                else: 
                    pairs['question'] = "I have information on two papers both published in Academic Medicine in 2015. For paper 1, "  + pairs['authors.1'] + ". " + pairs['affiliations.1'] + ". +  For Paper 2,   " + pairs['authors.2'] + ". " + pairs['affiliations.2'] +  "'. Which paper do you believe will have more views? Please respond either 'Paper 1.' or 'Paper 2.' followed by a short one sentence explanation."

            else: 
                if args.explanations == "False":
                    pairs['question'] = "I have the abstracts from two papers both published in Academic Medicine in 2015. Using only the information in the abstracts, which paper do you believe will have more views? PAPER 1 ABSTRACT: '" + pairs['Abstract.1'] + "' PAPER 2 ABSTRACT: '" + pairs['Abstract.2'] + "' Please respond either 'Paper 1' or 'Paper 2' with no additional text."
                else:
                    pairs['question'] = "I have the abstracts from two papers both published in Academic Medicine in 2015. Using only the information in the abstracts, which paper do you believe will have more views? PAPER 1 ABSTRACT: '" + pairs['Abstract.1'] + "' PAPER 2 ABSTRACT: '" + pairs['Abstract.2'] + "' Please respond either 'Paper 1.' or 'Paper 2.' followed by a short one-sentence explanation."

        # add which paper is a medical student, resident, medical school faculty, or a doctor more likely to click on? 
        elif args.question_type == "audience": 
            pairs['question'] = "I have information on two papers both published in Academic Medicine in 2015. Paper 1's title is: " + pairs['title.1'] + ". " + pairs['authors.1'] + ". " + pairs['affiliations.1'] + ". Paper 1's abstract is: '" +  pairs['Abstract.1'] +  "'. +  Paper 2's title is: " + pairs['title.2'] + ". " + pairs['authors.2'] + ". " + pairs['affiliations.2'] + ". Paper 2's abstract is: '" +  pairs['Abstract.2'] +  "'."
            pairs['question'] += " I would like you to decide which paper is more likely to be viewed by five different audiences. For each audience, please respond 'Paper 1' or 'Paper 2' depending on which paper you think that audience will be more likely to read."
            pairs['question'] += " The five audiences are people interested in medical school, medical students, medical residents, medical school faculty, and doctors. Your answer shold consist of 5 responses in order, separated by commas."
            pairs['question'] += " For example, if you think Paper 1 is more attractive to people interested in medical school, medical students, and residents, but Paper 2 is more attractive to medical school faculty or doctors, your response should be 'Paper 1, Paper 1, Paper 1, Paper 2, Paper 2' "

            if args.explanations == "False":
                pairs['question'] += " with no additional text. You must adhere to this format."
            else: 
                pairs['question'] += " followed by a one sentence explanation. You must adhere to this format."
        elif args.question_type == "qualities":
            pairs['question'] = "I have information on two papers both published in Academic Medicine in 2015. Paper 1's title is: " + pairs['title.1'] + ". " + pairs['authors.1'] + ". " + pairs['affiliations.1'] + ". Paper 1's abstract is: '" +  pairs['Abstract.1'] +  "'. +  Paper 2's title is: " + pairs['title.2'] + ". " + pairs['authors.2'] + ". " + pairs['affiliations.2'] + ". Paper 2's abstract is: '" +  pairs['Abstract.2'] +  "'."
            pairs['question'] += " I would like you to decide which paper best exhibits each one of these 10 qualities. The qualities are: topic novelty, topic popularity, title catchiness, generalizability, writing quality, meaningful contributions, author credibility, journal fit for Academic Medicine, collaboration scale, and applicability. "
            pairs['question'] += " For each quality, please respond 'Paper 1' or 'Paper 2' depending on which one you think best exhibits the quality. Keep in mind that these papers were published in 2015, and so your judgements should be based upon what was popular in 2015. Your answer should consist of 10 responses in order, each separated by commas."
            pairs['question'] += " For example, if you think Paper 1 had a more novel topic, Paper 1's topic was considered more popular at the time, Paper 1 had a more catchy title,Paper 2 is more generalizable to broader audiences, Paper 1 had better writing quality, Paper 2 had more meaningful contributions, Paper 1's authors were more credible, Paper 2 was a better fit for Academic Medicine, Paper 1 had more collaborators, and Paper 2's results were more applicable, then your response shold be 'Paper 1, Paper 1, Paper 1, Paper 2, Paper 1, Paper 2, Paper 1, Paper 2, Paper 1, Paper 2' "
            if args.explanations == "False":
                pairs['question'] += " with no additional text. You must adhere to this format."
            else: 
                pairs['question'] += " followed by a one sentence explanation. You must adhere to this format."
    
        for i in range(0, len(pairs)):
        #for i in range(0, 5): 
            question = pairs.iloc[i]['question']
            max_output_tokens = 16000
            try:
                response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are an AI expert in academic medical research."},
                    {"role": "user", "content": question},
                ],
                seed=args.seed,
                max_tokens=max_output_tokens,
            )
            except Exception as e:
                logging.exception(f"An error occurred for i = {i}")
            pairs.at[i, 'response'] = response.choices[0].message.content.strip()

            if i % 200 == 0:
                print(f"Done with index {i}")
                #pairs.to_csv("tweets_pairs_with_response.csv", index = False)
                pairs.to_csv(args.results_file, index = False)

        #pairs.to_csv("tweets_pairs_with_response.csv", index = False)
        pairs.to_csv(args.results_file, index = False)

    # multiple covariates instead 
    else:
        tweets_cleaned = tweets_cleaned.reset_index(drop = True)
        # assume for now that it's all types
        tweets_cleaned['question'] = "I have information on a paper published in Academic Medicine in 2015. The title is: " + tweets_cleaned['title'] + ". " + tweets_cleaned['authors'] + ". " + tweets_cleaned['affiliations'] + ". The abstract is: '" +  tweets_cleaned['Abstract'] + "."
        tweets_cleaned['question'] += " I would like you to give your opinion of 6 qualities of this paper. The qualities are: topic novelty, writing quality, meaningful contributions, author credibility, journal fit for Academic Medicine, and applicability. "
        tweets_cleaned['question'] += " For each quality, please give a rating from 1 to 10, where 10 means that this paper possesses this quality very strongly, and a 1 means this paper does not possess this quality at all."
        tweets_cleaned['question'] += " Your answer should be formatted as 6 numbers, separated by commas, with no additional text. For example, if your rating for topic novelty was 8, writing quality was a 2, meaningful contributions was a 9, author credibility was a 3, and applicability was a 5, you should respond '8, 2, 9, 3, 5' "
        if args.explanations == "False":
            tweets_cleaned['question'] += " with no additional text."
        else: 
            tweets_cleaned['question'] +=  " followed by a one sentence explanation of your reasoning."

        for i in range(0, len(tweets_cleaned)):
        #for i in range(0, 5): 
            question = tweets_cleaned.iloc[i]['question']
            max_output_tokens = 16000
            try:
                response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "You are an AI expert in academic medical research."},
                    {"role": "user", "content": question},
                ],
                seed=args.seed,
                max_tokens=max_output_tokens,
            )
            except Exception as e:
                logging.exception(f"An error occurred for i = {i}")
            tweets_cleaned.at[i, 'response'] = response.choices[0].message.content.strip()

            if i % 25 == 0:
                print(f"Done with index {i}")
                tweets_cleaned.to_csv(args.results_file, index = False)

        tweets_cleaned.to_csv(args.results_file, index = False)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="GPT model for aspect assignment (default: gpt-4o-mini)")
    parser.add_argument("--seed", type=int, default=93482)
    parser.add_argument("--results_file", type = str, required = True)
    parser.add_argument("--explanations", type = str, required = True, default = "False")
    parser.add_argument("--info_type", type = str, required = True, default = "all") # can either be "all", "abstract", "affiliations"
    parser.add_argument("--use_pairs", type = str, required = True, default = "False")
    parser.add_argument("--get_data", type = str, default = "False")
    parser.add_argument("--question_type", type = str, default = "basic") # cab be "basic" or "qualities"
    args = parser.parse_args()

    run_pipeline(args)