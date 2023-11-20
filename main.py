# Version 1.3
import os
import time
from collections import defaultdict
import re
import requests
import csv
os.environ['CAMELTOOLS_DATA'] = r'static/resources/camel_tools'

from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModel
import pickle
import torch
import faiss
import numpy as np
from flask import Flask
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.reinflector import Reinflector
from itertools import product
from camel_tools.utils.dediac import dediac_ar
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

arabic_stop_words = ['اللاتي', 'كلا', 'ولكن', 'عسى', 'بهما', 'فيما', 'لكي', 'لدى', 'لستما', 'كلما', 'نحن', 'سوى', 'الذي', 'كيتما', 'هم', 'هؤلاء', 'وهو', 'بين', 'ليس', 'هيت', 'أنتن', 'اللواتي', 'أيك', 'إذ', 'غير', 'كل', 'هذي', 'أولئك', 'هيا', 'بماذا', 'أولالك', 'فإذا', 'إليكما', 'ذان', 'هذا', 'ذينك', 'حاشا', 'ثمة', 'متى', 'بلى', 'ولا', 'أينما', 'على', 'كي', 'ليسوا', 'ذلكن', 'وما', 'أما', 'هناك', 'كأي', 'لها', 'ألا', 'لكم', 'اللذين', 'أولائك', 'ليستا', 'لو', 'أولئكم', 'عليه', 'فمن', 'إذن', 'فيه', 'ذانك', 'أقل', 'لهن', 'لستن', 'بهن', 'لاسيما', 'هنا', 'وإذ', 'هذين', 'بمن', 'بعد', 'ذي', 'أنتم', 'أيكما', 'لي', 'حيث', 'نحو', 'أنت', 'ماذا', 'إذما', 'إنها', 'إنه', 'أيكم', 'تي', 'تلكما', 'إلا', 'بس', 'حين', 'عندما', 'عن', 'كليهما', 'يا', 'كيفما', 'أولاء', 'أيها', 'ذو', 'أيا', 'به', 'كما', 'لهم', 'عند', 'إلينا', 'لسنا', 'ذلكما', 'أيهما', 'اللائي', 'بخ', 'كيت', 'بكم', 'هاك', 'بعض', 'الذين', 'كم', 'هيهات', 'أيهم', 'هاهنا', 'ولو', 'أنى', 'لكما', 'لكنما', 'وإذا', 'بكما', 'هو', 'إنما', 'سوف', 'لولا', 'كلاهما', 'لنا', 'أ', 'ما', 'لا', 'هل', 'ليسا', 'ته', 'تلك', 'اللتيا', 'فيها', 'هلا', 'ذا', 'أوه', 'ذه', 'اللتان', 'أي', 'عدا', 'فيم', 'هكذا', 'فلا', 'ذين', 'ها', 'أيه', 'إذا', 'دون', 'إن', 'لم', 'حبذا', 'أكثر', 'إليكم', 'ذوا', 'أف', 'إلى', 'كذا', 'بل', 'إلا أن', 'في', 'كأين', 'ومن', 'ذواتا', 'بنا', 'أيهن', 'عل', 'كليكما', 'أو', 'اللتين', 'ذاك', 'إليك', 'نعم', 'هنالك', 'خلا', 'أولائكم', 'كأن', 'كلتا', 'ذلك', 'هاته', 'هن', 'مع', 'منذ', 'كذلك', 'بما', 'لسن', 'أم', 'تلكم', 'أيكن', 'قد', 'هما', 'لستم', 'منها', 'بها', 'ذلكم', 'أن', 'بكن', 'مهما', 'كأنما', 'بك', 'اللذان', 'لوما', 'كيف', 'حتى', 'أين', 'بهم', 'تينك', 'ثم', 'له', 'من', 'ليست', 'شتان', 'أنا', 'ذواتي', 'لك', 'أنتما', 'مذ', 'لكن', 'مما', 'ريث', 'وإن', 'هذان', 'مه', 'بيد', 'فإن', 'حيثما', 'إما', 'لهما', 'إليكن', 'تين', 'بي', 'لكيلا', 'لن', 'عليك', 'لئن', 'ليت', 'لست', 'لما', 'منه', 'هي', 'ممن', 'هذه', 'لعل']
MAX_NUM_OF_EXAMPLES_PER_WORD = 5

SUPPORTED_EXAMPLES_TYPES = ["news","quraan","poetry","hadith"]
# SUPPORTED_EXAMPLES_TYPES = ["poetry","hadith","quraan"]

EXAMPLES_TYPES_REQUIRE_METADATA = ["quraan","hadith"]

API_KEY = ""

class MustashhedApp:
    def __init__(self, name):
        self.app = Flask(name)
        self.setup_routes()
        self.mode = "light"
        self.word = ""
        self.meaning = ""
        self.word_type = "Noun"
        self.resource_type = "all"
        self.examples = []
        self.list_of_meanings_dicts = []
        self.lemma_custom_checkbox = True
        self.setup = 2

        self.sentences = self.get_all_sentences()
        self.inverted_indices = self.get_all_inverted_indices()
        self.embeddings = self.get_all_embeddings()
        self.types_metadata = self.get_types_metadata()
        self.model, self.tokenizer = self.get_model_and_tokenizer()
        self.reinflector = Reinflector(MorphologyDB.builtin_db(flags='r'))
        print("CAMLTool's setup Completed")

    def setup_routes(self):
        self.app.add_url_rule('/', 'index', self.index, methods=['GET', 'POST'])
        self.app.add_url_rule('/get_examples_setup_1', 'get_examples_setup_1', self.get_examples_setup_1, methods=['GET', 'POST'])
        self.app.add_url_rule('/get_examples_setup_2', 'get_examples_setup_2', self.get_examples_setup_2, methods=['GET', 'POST'])
        self.app.add_url_rule('/change_the_setup', 'change_the_setup', self.change_the_setup, methods=['GET', 'POST'])
        self.app.add_url_rule('/set_mode', 'set_mode', self.set_mode, methods=['GET', 'POST'])
        self.app.add_url_rule('/contribute_in_alriyadh_dictionary_add', 'contribute_in_alriyadh_dictionary_add', self.contribute_in_alriyadh_dictionary_add, methods=['GET', 'POST'])
        self.app.add_url_rule('/contribute_in_alriyadh_dictionary_report', 'contribute_in_alriyadh_dictionary_report', self.contribute_in_alriyadh_dictionary_report, methods=['GET', 'POST'])


    def change_the_setup(self):
        try:
            self.setup = request.form.get('destination')
            return render_template(f'setup_{self.setup}.html',examples=[], word="", meaning="", mode=self.mode, word_type="Noun", resource_type = self.resource_type)
        except:
            return render_template(f'setup_1.html',examples=[], word="", meaning="", mode="light", word_type="Noun", resource_type = "all")

    def index(self):
        mode = request.form.get('mode')

        if mode == None:
            if self.mode != None:
                mode = self.mode
            else:
                mode = "light"
        if not self.setup:
            self.setup = 1
        return render_template(f'setup_{self.setup}.html',examples=[], word="", meaning="", mode=mode, word_type="Noun", resource_type = self.resource_type)

    def get_examples_setup_1(self):
        try:
            self.word = request.form.get('word')
            self.meaning = request.form.get('meaning')
            self.resource_type = request.form.get('resource_type')
            self.word_type = request.form['type-select']
            examples_list = []

            try:
                self.examples = self.get_examples_with_faiss(self.word, self.meaning, self.word_type, self.resource_type)

                # Sorting the examples based on the distance from the query (not by the type)
                for word in self.examples.keys():
                    self.examples[word] = sorted(self.examples[word], key=lambda x: x['distance'], reverse=False)

                for word in self.examples.keys():
                    examples = []
                    for dic in self.examples[word][:MAX_NUM_OF_EXAMPLES_PER_WORD]:
                        examples.append(dic['example'])
                    examples_list.append((word,examples))
                self.examples = examples_list
            except:
                print("Exception in get_examples_setup_1()")
                self.write_to_logs("Exception in get_examples_setup_1()")

            return render_template(f'setup_{self.setup}.html', examples=self.examples, word=self.word,meaning=self.meaning, word_type= self.word_type,resource_type = self.resource_type , mode=self.mode)
        except:
            return render_template(f'setup_1.html',examples=[], word="", meaning="", mode="light", word_type="Noun", resource_type = "all")

    def get_examples_setup_2(self):
        try:
            self.word = request.form.get('word')
            self.meaning = request.form.get('meaning')
            self.resource_type = request.form.get('resource_type')

            examples_list = []
            self.examples= examples_list

            if self.meaning == 'لعرض المعاني: قم بكتابة الكلمة ثم اضغط على "استرجاع المعاني"' or self.meaning == None:
                self.list_of_meanings_dicts = self.get_list_of_meanings_dicts_for_word(self.word)
                return render_template(f'setup_{self.setup}.html',list_of_meanings_dicts=self.list_of_meanings_dicts, examples=None, word=self.word,meaning=self.meaning, word_type= self.word_type,resource_type = self.resource_type , mode=self.mode)

            else:
                try:
                    self.meaning = request.form.get('meaning')
                    for item in self.list_of_meanings_dicts:
                        if self.meaning == item['meaning']: # becuase the meaning in the UI is consists from word_with_diacr+ space+meaning
                            meaning_dict = item
                            type = item['pos']
                            word_with_diacr = item['word_with_diacr']
                            if "V" in type:
                                self.word_type = "Verb"
                            elif "N" in type:
                                self.word_type = "Noun"
                            elif "R" in type:
                                self.word_type = "Preposition"
                            else:
                                self.word_type = "Noun"
                            break
                    self.meaning = word_with_diacr +": "+ self.meaning
                    self.examples = self.get_examples_with_faiss(self.word, self.meaning, self.word_type, self.resource_type)

                    # Sorting the examples based on the distance from the query (not by the type)
                    for word in self.examples.keys():
                        self.examples[word] = sorted(self.examples[word], key=lambda x: x['distance'], reverse=False)

                    for word in self.examples.keys():
                        examples = []
                        for dic in self.examples[word][:MAX_NUM_OF_EXAMPLES_PER_WORD]:
                            examples.append(dic['example'])
                        examples_list.append((word,examples))
                    self.examples = examples_list
                except:
                    self.write_to_logs("Exception in get_examples_setup_2()")

            return render_template(f'setup_{self.setup}.html',list_of_meanings_dicts=[meaning_dict], examples=self.examples, word=self.word,meaning="",resource_type = self.resource_type , mode=self.mode)
        except:
            return render_template(f'setup_1.html',examples=[], word="", meaning="", mode="light", word_type="Noun", resource_type = "all")

    def contribute_in_alriyadh_dictionary_add(self):
        try:
            word = request.form.get('word')
            meaning = request.form.get('meaning')
            resource_type = request.form.get('resource_type')
            word_type = request.form['type-select']
            example = request.form.get('example')

            self.add_data_to_csv("static/users_contributions.csv", word, example, meaning, resource_type, word_type, report=False, add=True)
            return render_template(f'setup_{self.setup}.html', word="",meaning="",resource_type = "",mode=self.mode)
        except:
            return render_template(f'setup_1.html',examples=[], word="", meaning="", mode="light", word_type="Noun", resource_type = "all")

    def contribute_in_alriyadh_dictionary_report(self):
        try:
            word = request.form.get('word')
            meaning = request.form.get('meaning')
            resource_type = request.form.get('resource_type')
            word_type = request.form['type-select']
            example = request.form.get('example')

            self.add_data_to_csv("static/users_contributions.csv", word, example, meaning, resource_type, word_type, report=True, add=False)
            return render_template(f'setup_{self.setup}.html', word="",meaning="",resource_type = "",mode=self.mode)
        except:
            return render_template(f'setup_1.html',examples=[], word="", meaning="", mode="light", word_type="Noun", resource_type = "all")


    def add_data_to_csv(self,file_path, word, example, meaning,resource_type, word_type, report=False, add=False):
        """
        Add data to a CSV file.

        Parameters:
        - file_path (str): The path to the CSV file.
        - word (str): The word to add.
        - example (str): An example of the word's usage.
        - meaning (str): The meaning of the word.
        - report (bool): Whether to report the word (default is False).
        - add (bool): Whether to add the word (default is False).
        """
        fieldnames = ["word", "example", "meaning", "resource_type", "word_type", "report", "add"]

        # Check if the file exists, and create it with header if not
        file_exists = False
        try:
            with open(file_path, 'r') as file:
                file_exists = True
        except FileNotFoundError:
            pass

        with open(file_path, 'a', newline='',encoding="UTF-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write header only if the file is newly created
            if not file_exists:
                writer.writeheader()

            # Write data
            writer.writerow({
                "word": word,
                "example": example,
                "meaning": meaning,
                "resource_type":resource_type,
                "word_type":word_type,
                "report": report,
                "add": add,

            })

    def get_list_of_meanings_dicts_for_word(self,word):
        url = "https://siwar.ksaa.gov.sa/api/alriyadh/search"
        query_param = word

        headers = {
            "apikey": API_KEY
        }

        params = {
            "query": query_param,
        }

        # Make the GET request with certificate verification
        response = requests.get(url, headers=headers, params=params, verify=False)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse and work with the response data (assuming it's in JSON format)
            data = response.json()

            list_of_meanings_dicts = []
            id = 0
            for i in range(len(data)):
                if len(data[i]['senses'])>0:
                    for dic in data[i]['senses']:
                        meaning_dict = {}
                        meaning_dict["id"] = id
                        meaning_dict["meaning"] = str(dic['definition']['textRepresentations'][0]['form'])
                        meaning_dict['nonDiacriticsLemma'] = str(data[i]['nonDiacriticsLemma'])
                        meaning_dict["word_with_diacr"] = str(data[i]['nonDiacriticsLemma'])
                        meaning_dict["pos"] = str(data[i]['pos'])
                        if len(meaning_dict["meaning"]) > 0:
                            list_of_meanings_dicts.append(meaning_dict)
                            id+=1

        return list_of_meanings_dicts

    def set_mode(self):
        try:
            mode = request.form.get('mode')
            self.mode = mode
            return render_template(f'setup_{self.setup}.html', examples=[], word="",meaning="", word_type= "Noun", mode=self.mode,resource_type = "all")
        except:
            return render_template(f'setup_1.html',examples=[], word="", meaning="", mode="light", word_type="Noun", resource_type = "all")

    def move_tuple_to_beginning(self,lst, target_value):
        for i, item in enumerate(lst):
            if item[0] == target_value:
                # Swap the current tuple with the first tuple
                lst[0], lst[i] = lst[i], lst[0]
                break

    def get_model_and_tokenizer(self):
        model_name = "static/resources/bert-base-arabertv02"

        if os.path.exists(model_name) and os.path.isdir(model_name):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        else:
            model_name = "aubmindlab/bert-base-arabertv02"

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

            tokenizer.save_pretrained("static/resources/bert-base-arabertv02")
            model.save_pretrained("static/resources/bert-base-arabertv02")


        return model.to(device), tokenizer


    def get_all_inverted_indices(self):
        inverted_indices = {}
        for _type in tqdm(SUPPORTED_EXAMPLES_TYPES,desc="get_all_inverted_indices()"):
            # Load the list from the file
            with open(f"static/resources/{_type}/inverted_index.pkl", 'rb') as file:
                inverted_indices[_type] = pickle.load(file)
        return inverted_indices

    def get_types_metadata(self):
        types_metadata = {}
        current_types_require_metadata = [_type for _type in EXAMPLES_TYPES_REQUIRE_METADATA if _type in SUPPORTED_EXAMPLES_TYPES]
        for _type in tqdm(current_types_require_metadata,desc="get_types_metadata()"):
            with open(f"static/resources/{_type}/{_type}_index_to_meta_data_dict.pkl", 'rb') as file:
                types_metadata[_type] = pickle.load(file)
        return types_metadata

    def get_all_sentences(self):
        sentences = {}
        for _type in tqdm(SUPPORTED_EXAMPLES_TYPES,desc="get_all_sentences()"):
            # Load the list from the file
            with open(f"static/resources/{_type}/Processed_sentences.pkl", 'rb') as file:
                sentences[_type] = pickle.load(file)

        return sentences

    def get_all_embeddings(self):
        embeddings = {}
        for _type in tqdm(SUPPORTED_EXAMPLES_TYPES,desc="get_all_embeddings()"):
            loaded_tensor_dict = torch.load(f"static/resources/{_type}/embeddings.pt",map_location=device)
            embeddings_list = []
            # Iterate over the dictionary values (tensors) and add them to the list
            for tensor_list in loaded_tensor_dict.values():
                for tensor in tensor_list:
                    embeddings_list.append(tensor)
            embeddings[_type] = embeddings_list

        return embeddings

    # def make_queries_embeddings(self, words, meaning, word_type):
    #     query = f"{words[0]}:{meaning}" # Make only one query for the first word form .. as we don't need to make query for each word form
    #     input_ids = self.tokenizer(query, return_tensors="pt", padding=True).to(device)
    #     # Get the model's output
    #     with torch.no_grad():
    #         output = self.model(**input_ids)
    #
    #     # Get the embeddings from the output
    #     queries_embeddings = output.last_hidden_state
    #     # Get the Average over the 1st dim
    #     queries_embeddings = torch.mean(queries_embeddings, dim=1)
    #
    #     return [queries_embeddings for word in words]

    def make_queries_embeddings(self, words, meaning, word_type):
        queries = [f"{word}:{meaning}" for word in words]
        input_ids = self.tokenizer(queries, return_tensors="pt", padding=True).to(device)
        # Get the model's output
        with torch.no_grad():
            output = self.model(**input_ids)

        # Get the embeddings from the output
        queries_embeddings = output.last_hidden_state
        # Get the Average over the 1st dim
        queries_embeddings = torch.mean(queries_embeddings, dim=1)

        if len(words) == 1:
            return [queries_embeddings]
        else:
            return torch.split(queries_embeddings, 1, dim=0)

    def remove_near_distances(self,distances, indices, threshold):
        new_distances = []
        new_indices = []

        for i in range(len(distances)):
            current_distance = distances[i]
            should_add = True

            # Check if the current distance is near any existing distance
            for j in range(len(new_distances)):
                existing_distance = new_distances[j]

                if abs(current_distance - existing_distance) < threshold:
                    should_add = False
                    break

            if should_add:
                new_distances.append(current_distance)
                new_indices.append(indices[i])

        return new_distances, new_indices

    def get_examples_with_faiss(self,word ,meaning, word_type, resource_type):

        meaning_filtered_words = [word for word in meaning.split() if word not in arabic_stop_words]
        # Join the filtered words back into a sentence
        meaning = ' '.join(meaning_filtered_words)

        print(f"Query: word:{word} ,meaning:{meaning}, word_type:{word_type}, resource_type:{resource_type}")
        self.write_to_logs(f"Query: word:{word} ,meaning:{meaning}, word_type:{word_type}, resource_type:{resource_type}")

        lemma_custom_checkbox = True
        examples_dict = {}
        current_resource_types = []
        if resource_type == "all":
            for _type in SUPPORTED_EXAMPLES_TYPES:
                current_resource_types.append(_type)
        else:
            current_resource_types.append(resource_type)

        for _type in current_resource_types:
            start_time = time.time()
            all_forms_of_word = self.get_all_forms_of_word(_type, lemma_custom_checkbox, word, word_type)
            print(all_forms_of_word)
            self.write_to_logs(all_forms_of_word)

            if len(all_forms_of_word) > 0:
                queries_embeddings = self.make_queries_embeddings(all_forms_of_word, meaning, word_type)
            else:
                continue
            current_resource_type_all_words_forms_examples = []
            for word_form, query_embedding in zip(all_forms_of_word,queries_embeddings):
                words_form_examples = []

                # Filter the embeddings to contain only the embeddings of the word (to search only with those embeddings)
                inverted_indicies = self.inverted_indices[_type][word_form]
                faiss_index_to_sentence_index_dict = {faiss_output_index:sentence_index for faiss_output_index,sentence_index in enumerate(inverted_indicies)}
                embeddings = []
                for index in inverted_indicies:
                    embeddings.append(self.embeddings[_type][index])
                # # Stack the tensors into a single tensor
                filtered_embeddings = torch.stack(embeddings)

                # Ensure that (filtered_embeddings) is a 2D array of shape (num_embeddings, embedding_dim)
                embedding_np = filtered_embeddings.cpu().numpy()

                # Create a Faiss index
                index = faiss.IndexFlatL2(embedding_np.shape[1])  # L2 (Euclidean) distance index

                # Add the embeddings to the Faiss index
                index.add(embedding_np)

                # Define the number of nearest neighbors you want to retrieve
                k = len(self.inverted_indices[_type][word_form])

                # Define a query embedding for which you want to find similar embeddings
                query_embedding = np.array(query_embedding.cpu(), dtype=np.float32)  # Replace with your query embedding

                # Perform a search to find the nearest neighbors
                distances, indices = index.search(query_embedding, k)

                new_distances, new_indices = self.remove_near_distances(distances=distances.tolist()[0], indices = indices.tolist()[0], threshold = 3)
                faiss_indices = new_indices
                distances = []
                for faiss_index ,distance in zip(faiss_indices,new_distances):
                    sentence_index = faiss_index_to_sentence_index_dict[faiss_index]
                    words_form_examples.append(self.generate_html_sentence(sentence=self.sentences[_type][sentence_index], word_to_highlight=word_form,example_type=_type,sentence_index=sentence_index))
                    self.write_to_logs(f"word:{word_form}, example:{self.sentences[_type][sentence_index]}")
                    distances.append(distance)
                current_resource_type_all_words_forms_examples.append((word_form, words_form_examples, distances))
            print("--- %s seconds ---" % (time.time() - start_time))

            examples_dict[_type] = current_resource_type_all_words_forms_examples

        # Create a defaultdict with default values as empty lists
        words_examples = defaultdict(list)

        for examples_type in examples_dict.keys():
            for word_examples_distances_tuple in examples_dict[examples_type]:
                word, examples, distances = word_examples_distances_tuple
                for example,distance in zip(examples, distances):
                    words_examples[word].append({"example":example,"distance":distance,"type":examples_type})

        return words_examples

    def extract_window_around_word(self, text, target_word, window_size=8):
        # Split the text into words
        words = text.split()

        # Find the index of the target word
        try:
            word_index = words.index(target_word)
        except:
            # Target word not found in the text
            return text

        # Calculate the start and end indices for the window
        start_index = max(0, word_index - window_size)
        end_index = min(len(words), word_index + window_size + 1)

        # Extract the window around the target word
        window = words[start_index:end_index]

        # Join the words in the window to form the final text
        result_text = ' '.join(window)

        if start_index != 0:
            result_text = "..." + result_text

        if end_index != len(text)-1:
            result_text = result_text +"..."

        return result_text

    def get_all_forms_of_word(self, _type, lemma_custom_checkbox, word, word_type):
        all_forms_of_word = []
        all_forms_of_word.append(word)
        if lemma_custom_checkbox:
            forms_of_word = self.get_all_forms_of_arabic_word(word, word_type)
            for word_form in forms_of_word:
                all_forms_of_word.append(word_form)
        # Add non-dediac words also
        all_forms_of_word = all_forms_of_word + [dediac_ar(word_form) for word_form in all_forms_of_word]
        all_forms_of_word = list(set(all_forms_of_word))
        all_forms_of_word = [word_form for word_form in all_forms_of_word if
                             len(self.inverted_indices[_type][word_form]) > 0]
        if word in all_forms_of_word:
            all_forms_of_word.remove(word)
            all_forms_of_word.insert(0, word)
        return all_forms_of_word

    def generate_html_sentence(self, sentence, word_to_highlight,example_type,sentence_index):
        orignal_sentence = sentence
        sentence = self.extract_window_around_word(sentence,word_to_highlight)
        if example_type == "quraan":
            meta_data = self.types_metadata['quraan'][sentence_index]
            sentence = "﴿ "+sentence+" ﴾" + f" [{meta_data['sura']}:{meta_data['aya']}]"
            orignal_sentence = "﴿ "+orignal_sentence+" ﴾" + f" [{meta_data['sura']}:{meta_data['aya']}]"
            # sentence = "﴾"+sentence+"﴿"+ "[" + "سورة: " + meta_data['sura'] + " آية: " + str(meta_data['aya'])

        if example_type == "hadith":
            meta_data = self.types_metadata['hadith'][sentence_index]
            sentence = "("+sentence+")"+ f" [{meta_data}]"
            orignal_sentence = "("+orignal_sentence+")"+ f" [{meta_data}]"

        pattern_arabic_with_diacritics = re.compile(r'[\u0600-\u06FF]+')
        matches_arabic = pattern_arabic_with_diacritics.finditer(sentence)

        # Initialize the words_positions_in_sentences as a defaultdict of lists
        words_positions_in_sentences = defaultdict(list)

        for match in matches_arabic:
            start_pos = match.start()
            end_pos = match.end()
            match_text = match.group(0)
            words_positions_in_sentences[match_text].append((start_pos, end_pos))

        red_words = words_positions_in_sentences[word_to_highlight]

        html_output = f'<span title="{orignal_sentence}">'


        current_pos = 0
        for start, end in red_words:
            html_output += sentence[current_pos:start]  # Append non-red part
            html_output += f'<span class="word_to_highlight">{sentence[start:end]}</span>'  # Append red part with underline
            current_pos = end

        html_output += sentence[current_pos:]  # Append the remaining non-red part

        if example_type == "quraan":
            html_output += '<span title="موقع تنزيل" style="font-size: 14px;margin-right: 6px;">مصدر النص القرآني : <a href="https://tanzil.net/" style="color: Black;">موقع تنزيل</a></span>'

        html_output += '</span>'
        return html_output

    def get_all_forms_of_arabic_word(self,word, word_type):
        all_forms = []

        """
        asp - Aspect
            c - Command
            i - Imperfective
            p - Perfective
            na - Not applicable
        
        gen - Gender
            f - Feminine
            m - Masculine
            na - Not applicable
        
        mod - Mood
            i - Indicative
            j - Jussive
            s - Subjunctive
            na - Not applicable
            u - Undefined
        
        num - Number
            s - Singular
            d - Dual
            p - Plural
            na - Not applicable
            u - Undefined
        
        pos - Part-of-speech
            noun - Noun
            adj - Adjective
            pron - Pronoun
            verb - Verb
            part - Particle
            prep - Preposition
            abbrev - Abbreviation
            punc - Punctuation
            conj - Conjunction
        """
        pos = {
            "Noun": ['noun',  'adj', 'pron'],
            "Verb": ['verb'],
            "Preposition": ['prep']
        }
        features = {
            'asp': ['p', 'i', 'c', 'na'],
            'gen': ['f', 'm', 'na', 'u'],
            'mod': ['i', 'j', 's', 'na', 'u'],
            'num': ['s', 'd', 'p', 'na', 'u'],
            'pos': pos[word_type],
        }

        # Create combinations of feature values
        feature_combinations = product(*(features[feature] for feature in features))

        for feature_values in feature_combinations:
            feature_dict = dict(zip(features, feature_values))
            analyses = self.reinflector.reinflect(word, feature_dict)
            unique_diacritizations = set(a['diac'] for a in analyses)
            all_forms.extend(unique_diacritizations)
        return all_forms


    def write_to_logs(self,data):
        # Open the file in append mode ('a' flag) to create if it doesn't exist and append if it does
        with open('logs.txt', 'a',encoding='UTF-8') as file:
            # Write the input data to the file with a newline character
            file.write(str(data) + '\n')


if __name__ == '__main__':
    my_app = MustashhedApp(__name__)
    my_app.app.run(debug=False,host="0.0.0.0")