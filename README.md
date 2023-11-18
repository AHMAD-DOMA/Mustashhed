# Mustashhed - Arabicthon 2023 Project

Welcome to "Mustashhed," our open-source project submitted as a contribution to Arabicthon 2023 (https://arabicthon.ksaa.gov.sa/). Mustashhed is a Flask web application accessible at https://mustashhed.ngrok.dev/.

## Idea of the Project

Our project leverages a set of large datasets of text containing a substantial number of Arabic texts. We utilize [pySBD](https://github.com/nipunsadvilkar/pySBD) (Python Sentence Boundary Disambiguation) to split the text from the corpora into sentences, forming the examples.

Next, we employ the [aubmindlab/bert-base-arabertv02](https://huggingface.co/aubmindlab/bert-base-arabertv02) model to obtain embeddings for these examples. When a user enters a word along with its meaning and type, or uses the [Alriyadh dictionary](https://dictionary.ksaa.gov.sa/), we utilize [camel_tools](https://github.com/CAMeL-Lab/camel_tools) to retrieve all forms of the word based on its type (verb, noun, preposition).

For each word form, it is associated with the meaning to create a query. This query is then passed to the embedding model. Subsequently, we utilize [Faiss](https://github.com/facebookresearch/faiss), a library for efficient similarity search and clustering of dense vectors, to find the most similar examples to the user's query.

## Getting Started

To run and host this project locally, follow these steps:

1. Download the "resources" folder from [this Google Drive link](https://drive.google.com/drive/folders/1mqNd3l0jy19Nwaj5DV9v7dlHYL0hklkP?usp=sharing).
2. Replace the existing "static/resources" folder in the project with the downloaded one.
3. Create a new python virtual environment, and activate it.
4. Run the following command:
```bash
pip install -r requirements.txt
```

4. Follow the installation steps in the [CAMeL Tools GitHub repository](https://github.com/CAMeL-Lab/camel_tools) to install the camel-tools package.
5. Run the following command:

```bash
pip install faiss-cpu
```
6. If you want to use 'Alriyadh dictionary' meanings you need to ask them (dictionary@ksaa.gov.sa) to give you an API_KEY, then if they porvided you with the key, put your key in the API_KEY variable in main.py.
7. Run the main application.

```bash
python main.py
```

**Note:** The first time you run the application, it may take some time to download the `aubmindlab/bert-base-arabertv02` model. Subsequent runs will not require this download.

## Data Resources

### Quraan
- **Tanzil Documents**: [Tanzil Documents](https://tanzil.net/docs/)

### Hadith
- **Open Hadith Library**: [mhashim6/Open-Hadith-Data](https://github.com/mhashim6/Open-Hadith-Data)
  - Databases of 9 different books, including the Six Books.
  - Data source: [ceefour/hadith-islamware](https://github.com/ceefour/hadith-islamware)

### News
- **SANAD Dataset**: [SANAD Dataset](https://data.mendeley.com/datasets/57zpx667y9)

### Poetry
- **Poetry Documents**: [Poetry Documents](https://drive.google.com/file/d/16jr56LBhKuZGYXZi_A39ab4GCGxZa-X_/view?usp=sharing)
- Data sources:
  - Shamela website: [Shamela.ws](https://shamela.ws/)
  - Adab website: [Adab.com](https://www.adab.com/)
  - Aldiwan website: [Aldiwan.net](https://www.aldiwan.net/)
