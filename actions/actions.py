from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

import pickle
import numpy
import pandas as pd
import nltk
from sklearn.preprocessing import StandardScaler
import re
from collections import Counter
import textstat
from lexicalrichness import LexicalRichness
from nltk.corpus import stopwords


class ActionHelloWorld(Action):
    stop_words = set(stopwords.words('english'))

    def name(self) -> Text:
        return "action_analyze"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        svc_model = pickle.load(open("/Users/emilioverde/Desktop/rasa-init-demo/actions/saved_model", 'rb'))
        dataset = pd.DataFrame(columns=['title', 'text', 'source'], index=range(1))

        input = tracker.latest_message['text']

        dataset["text"].iloc[0] = input
        dataset["title"].iloc[0] = 'missing'
        dataset["source"].iloc[0] = 'missing'

        dataset.text.fillna(dataset.title, inplace=True)
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        dataset.title.fillna('missing', inplace=True)

        dataset['title_num_uppercase'] = dataset['title'].str.count(r'[A-Z]')
        dataset['text_num_uppercase'] = dataset['text'].str.count(r'[A-Z]')
        dataset['text_len'] = dataset['text'].str.len()
        dataset['text_pct_uppercase'] = dataset.text_num_uppercase.div(dataset.text_len)

        from nltk.corpus import stopwords

        stop_words = set(stopwords.words('english'))
        dataset['title_num_stop_words'] = dataset['title'].str.split().apply(lambda x: len(set(x) & stop_words))
        dataset['text_num_stop_words'] = dataset['text'].str.split().apply(lambda x: len(set(x) & stop_words))
        dataset['text_word_count'] = dataset['text'].apply(lambda x: len(str(x).split()))
        dataset['text_pct_stop_words'] = dataset['text_num_stop_words'] / dataset['text_word_count']

        dataset.drop(['text_num_uppercase', 'text_len', 'text_num_stop_words', 'text_word_count'], axis=1, inplace=True)
        dataset['token'] = dataset.apply(lambda row: nltk.word_tokenize(row['title']), axis=1)
        dataset['pos_tags'] = dataset.apply(lambda row: nltk.pos_tag(row['token']), axis=1)
        tag_count_dataset = pd.DataFrame(dataset['pos_tags'].map(lambda x: Counter(tag[1] for tag in x)).to_list())
        dataset = pd.concat([dataset, tag_count_dataset], axis=1).fillna(0).drop(['pos_tags', 'token'], axis=1)

        dataset = dataset[
            ['title', 'text', 'source', 'title_num_uppercase', 'text_pct_uppercase', 'title_num_stop_words',
             'text_pct_stop_words']].rename(columns={'NNP': 'NNP_title'})

        dataset['token'] = dataset.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
        dataset['pos_tags'] = dataset.apply(lambda row: nltk.pos_tag(row['token']), axis=1)

        print(len(dataset.columns))
        tag_count_dataset = pd.DataFrame(dataset['pos_tags'].map(lambda x: Counter(tag[1] for tag in x)).to_list())
        print(tag_count_dataset)
        original_tags = ['NNS', 'VBP', 'IN', 'DT', 'JJ', 'NN', 'NNP', 'MD', 'VB', 'PRP', 'WRB', 'TO', ',', 'VBZ', 'WDT',
                         'CC', '.', 'PRP$', 'RB', 'VBG', 'VBD', 'CD', 'WP', 'RBR', 'VBN', 'JJS', 'RP', 'JJR', ':', '``',
                         '(', ')', "''", 'POS', 'EX', 'PDT', 'RBS', 'NNPS', '$', 'WP$', 'FW', 'UH', '#', 'SYM', '.']

        df_tags = pd.DataFrame(columns=original_tags, index=range(1))

        for tag in original_tags:
            if tag in tag_count_dataset.columns:
                df_tags[tag] = tag_count_dataset[tag]
            else:
                df_tags[tag] = numpy.zeros(1).tolist()

        dataset = pd.concat([dataset, df_tags], axis=1).fillna(0).drop(['pos_tags', 'token'], axis=1)

        dataset['num_negation'] = dataset['text'].str.lower().str.count(
            "no|not|never|none|nothing|nobody|neither|nowhere|hardly|scarcely|barely|doesn’t|isn’t|wasn’t|shouldn’t|wouldn’t|couldn’t|won’t|can't|don't")
        dataset['num_interrogatives_title'] = dataset['title'].str.lower().str.count(
            "what|who|when|where|which|why|how")
        dataset['num_interrogatives_text'] = dataset['text'].str.lower().str.count("what|who|when|where|which|why|how")

        # TRAINING MODELLO
        reading_ease = []
        for doc in dataset['text']:
            reading_ease.append(textstat.flesch_reading_ease(doc))

        smog = []
        for doc in dataset['text']:
            smog.append(textstat.smog_index(doc))

        kincaid_grade = []
        for doc in dataset['text']:
            kincaid_grade.append(textstat.flesch_kincaid_grade(doc))

        liau_index = []
        for doc in dataset['text']:
            liau_index.append(textstat.coleman_liau_index(doc))

        readability_index = []
        for doc in dataset['text']:
            readability_index.append(textstat.automated_readability_index(doc))

        readability_score = []
        for doc in dataset['text']:
            readability_score.append(textstat.dale_chall_readability_score(doc))

        difficult_words = []
        for doc in dataset['text']:
            difficult_words.append(textstat.difficult_words(doc))

        write_formula = []
        for doc in dataset['text']:
            write_formula.append(textstat.linsear_write_formula(doc))

        gunning_fog = []
        for doc in dataset['text']:
            gunning_fog.append(textstat.gunning_fog(doc))

        text_standard = []
        for doc in dataset['text']:
            text_standard.append(textstat.text_standard(doc))

        dataset['flesch_reading_ease'] = reading_ease
        dataset['smog_index'] = smog
        dataset['flesch_kincaid_grade'] = kincaid_grade
        dataset['automated_readability_index'] = readability_index
        dataset['dale_chall_readability_score'] = readability_score
        dataset['difficult_words'] = difficult_words
        dataset['linsear_write_formula'] = write_formula
        dataset['gunning_fog'] = gunning_fog
        dataset['text_standard'] = text_standard

        ttr = []
        for doc in dataset['text']:
            lex = LexicalRichness(doc)
            ttr.append(lex.ttr)

        dataset['ttr'] = ttr
        print(len(dataset.columns))

        dataset['num_powerWords_text'] = dataset['text'].str.lower().str.count(
            'improve|trust|immediately|discover|profit|learn|know|understand|powerful|best|win|more|bonus|exclusive|extra|you|free|health|guarantee|new|proven|safety|money|now|today|results|protect|help|easy|amazing|latest|extraordinary|how to|worst|ultimate|hot|first|big|anniversary|premiere|basic|complete|save|plus|create')
        dataset['num_casualWords_text'] = dataset['text'].str.lower().str.count(
            'make|because|how|why|change|use|since|reason|therefore|result')
        dataset['num_tentativeWords_text'] = dataset['text'].str.lower().str.count(
            'may|might|can|could|possibly|probably|it is likely|it is unlikely|it is possible|it is probable|tends to|appears to|suggests that|seems to')
        dataset['num_emotionWords_text'] = dataset['text'].str.lower().str.count(
            'ordeal|outrageous|provoke|repulsive|scandal|severe|shameful|shocking|terrible|tragic|unreliable|unstable|wicked|aggravate|agony|appalled|atrocious|corruption|damage|disastrous|disgusted|dreadatasetul|eliminate|harmful|harsh|inconsiderate|enraged|offensive|aggressive|frustrated|controlling|resentful|anger|sad|fear|malicious|infuriated|critical|violent|vindictive|furious|contrary|condemning|sarcastic|poisonous|jealous|retaliating|desperate|alienated|unjustified|violated')

        def cleantext(string):
            text = string.lower().split()
            text = " ".join(text)
            text = re.sub(r"http(\S)+", ' ', text)
            text = re.sub(r"www(\S)+", ' ', text)
            text = re.sub(r"&", ' and ', text)
            text = text.replace('&amp', ' ')
            text = re.sub(r"[^0-9a-zA-Z]+", ' ', text)
            text = text.split()
            text = [w for w in text if not w in stop_words]
            text = " ".join(text)
            return text

        dataset['text'] = dataset['text'].map(lambda x: cleantext(x))
        dataset['title'] = dataset['title'].map(lambda x: cleantext(x))
        dataset['source'] = dataset['source'].map(lambda x: cleantext(x))

        classes = {"TRUE": 1, "FAKE": 0}

        def correlation(dataset, threshold):
            col_corr = set()
            corr_matrix = dataset.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] >= threshold and (corr_matrix.columns[j] not in col_corr):
                        colname = corr_matrix.columns[i]
                        col_corr.add(colname)
                        if colname in dataset.columns:
                            del dataset[colname]

        scaler = StandardScaler()
        dataset = dataset.drop(['title', 'text', 'source', 'text_standard'], axis=1)

        testset = dataset.to_numpy()

        X_test = scaler.fit_transform(testset)

        svc_model.random_state=0
        pred = svc_model.predict(X_test)

        if pred == 1:
            dispatcher.utter_message("notizia vera")
            return []
        elif pred == 0:
            dispatcher.utter_message("alert: fake")
            return []
