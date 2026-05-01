"""Data Loading Utilities."""
import logging
import os
import json
import hashlib
import datasets


def load_ds(dataset_name, seed, add_options=None):
    """Load dataset."""
    train_dataset, validation_dataset = None, None
    if dataset_name == "squad":
        dataset = datasets.load_dataset("squad_v2")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == 'svamp':
        dataset = datasets.load_dataset('ChilleD/SVAMP')

        train_dataset = dataset["train"]
        validation_dataset = dataset["test"]

        reformat = lambda x: {
            'question': x['Question'], 'context': x['Body'], 'type': x['Type'],
            'equation': x['Equation'], 'id': x['ID'],
            'answers': {'text': [str(x['Answer'])]}}

        train_dataset = [reformat(d) for d in train_dataset]
        _validation_dataset = [reformat(d) for d in validation_dataset]
        # For semantic entropy generation: merge training with test set for more samples.
        validation_dataset = _validation_dataset + train_dataset

    elif dataset_name == 'nq':
        dataset = datasets.load_dataset("nq_open")
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        reformat = lambda x: {
            'question': x['question']+'?',
            'answers': {'text': x['answer']},
            'context': '',
            'id': md5hash(str(x['question'])),
        }

        train_dataset = [reformat(d) for d in train_dataset]
        validation_dataset = [reformat(d) for d in validation_dataset]

    elif dataset_name == "trivia_qa":
        dataset = datasets.load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')['unmodified']
       
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']
    
    elif dataset_name == "med_qa":
        dataset = datasets.load_dataset("bigbio/med_qa")
        logging.info('Dataset: %s', dataset)
        for key in 'train', 'validation':
            ids = ['train' + str(i) for i in range(len(dataset[key]))]
            dataset[key] = dataset[key].add_column("id", ids)

            new_column = [None] * len(dataset[key])
            dataset[key] = dataset[key].add_column("context", new_column)

            answers = [
                {'text': [answer], 'answer_start': [0]}
                for answer in dataset[key][:]['answer']
            ]
            dataset[key] = dataset[key].add_column("answers", answers)

            if add_options:
                options = dataset[key][:]['options']
                options_string = [
                    [option['value'] + '\n' for option in option_list]
                    for option_list in options
                ]
                questions = dataset[key][:]['question']
                # zip questions and options
                questions_options = [
                    question + '\n' + ''.join(option_list)
                    for question, option_list in zip(questions, options_string)
                ]

                dataset[key] = dataset[key].remove_columns(['question'])
                dataset[key] = dataset[key].add_column(
                    "question", questions_options)

        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"]

    elif dataset_name == "bioasq":
        # http://participants-area.bioasq.org/datasets/ we are using training 11b
        # could also download from here https://zenodo.org/records/7655130
        # scratch_dir = os.getenv('SCRATCH_DIR', '.')
        path = "~/uncertainty/data/bioasq/training11b.json"
        with open(path, "rb") as file:
            data = json.load(file)

        questions = data["questions"]
        dataset_dict = {
            "question": [],
            "answers": [],
            "id": []
        }

        for question in questions:
            if "exact_answer" not in question:
                continue
            dataset_dict["question"].append(question["body"])
            if "exact_answer" in question:

                if isinstance(question['exact_answer'], list):
                    exact_answers = [
                        ans[0] if isinstance(ans, list) else ans
                        for ans in question['exact_answer']
                    ]
                else:
                    exact_answers = [question['exact_answer']]

                dataset_dict["answers"].append({
                    "text": exact_answers,
                    "answer_start": [0] * len(question["exact_answer"])
                })
            else:
                dataset_dict["answers"].append({
                    "text": question["ideal_answer"],
                    "answer_start": [0]
                })
            dataset_dict["id"].append(question["id"])

            dataset_dict["context"] = [None] * len(dataset_dict["id"])

        dataset = datasets.Dataset.from_dict(dataset_dict)

        # split into training and validation set
        dataset = dataset.train_test_split(test_size=0.8, seed=seed)
        train_dataset = dataset['train']
        validation_dataset = dataset['test']

    elif dataset_name == "xsum":
        # XSum (extreme summarization) — BBC news articles + single-sentence summaries.
        # Used as the Lookback Gate testbed because generated summaries are long enough
        # (50-100 tokens) for attention drift to manifest across generation steps.
        dataset = datasets.load_dataset("EdinburghNLP/xsum")

        def reformat_xsum(x):
            # Store the article in 'question' so the existing prompt builders work.
            # The article prefix and "Summary:" suffix are injected here so that
            # run_qa_generation.py sees a BRIEF_PROMPT + "question\nAnswer:" which maps
            # naturally to "article\nSummary:" when XSUM_BRIEF_PROMPT is used.
            return {
                'question': f"Article: {x['document']}\nSummary:",
                'answers': {'text': [x['summary']]},
                'context': x['document'],
                'id': str(x['id']),
            }

        train_dataset = [reformat_xsum(d) for d in dataset["train"]]
        validation_dataset = [reformat_xsum(d) for d in dataset["validation"]]

    elif dataset_name == "ragtruth":
        # RAGTruth — RAG hallucination benchmark with ground-truth labels.
        # Stages 1+2 (generation + NLI) are handled by map_ragtruth_labels.py;
        # this branch exists for reference and for any tooling that calls load_ds.
        dataset = datasets.load_dataset("wandb/RAGTruth-processed")
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        def reformat_ragtruth(x):
            labels = x.get('hallucination_labels_processed', {}) or {}
            hallucinated = int(
                labels.get('evident_conflict', 0) == 1 or
                labels.get('baseless_info', 0) == 1
            )
            return {
                'question':    x['query'],
                'context':     x['context'],
                'answers':     {'text': [x['output']]},
                'id':          str(x.get('id', md5hash(x['query']))),
                'hallucinated': hallucinated,
                'input_str':   x.get('input_str', ''),
            }

        def is_llama2(x):
            return 'llama-2-7b-chat' in str(x.get('model', '')).lower()

        available_splits = list(dataset.keys())
        train_split = 'train' if 'train' in available_splits else available_splits[0]
        val_split   = 'test'  if 'test'  in available_splits else available_splits[-1]
        train_dataset      = [reformat_ragtruth(d) for d in dataset[train_split] if is_llama2(d)]
        validation_dataset = [reformat_ragtruth(d) for d in dataset[val_split]   if is_llama2(d)]

    elif dataset_name == "halueval_qa":
        # HaluEval QA — questions with a correct answer and supporting knowledge.
        # We load the parquet directly via hf_hub_download because the datasets
        # library loader for pminervini/HaluEval has a broken config (name=None).
        # We use the 'qa/data' split which has separate right_answer / hallucinated_answer.
        import pandas as pd
        from huggingface_hub import hf_hub_download
        md5hash = lambda s: str(int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16))

        parquet_path = hf_hub_download(
            repo_id="pminervini/HaluEval",
            filename="qa/data-00000-of-00001.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(parquet_path)

        def reformat_halueval(row):
            return {
                'question': row['question'],
                'answers': {'text': [row['right_answer']]},
                'context': row.get('knowledge', ''),
                'id': md5hash(row['question']),
            }

        all_data = [reformat_halueval(df.iloc[i]) for i in range(len(df))]
        split_ds = datasets.Dataset.from_list(all_data).train_test_split(
            test_size=0.2, seed=seed)
        train_dataset = split_ds['train']
        validation_dataset = split_ds['test']

    elif dataset_name == "cnn_dailymail":
        # CNN/DailyMail summarization — same structure as XSum.
        # We store the article in 'context' and the highlight summary in 'answers'.
        dataset = datasets.load_dataset("cnn_dailymail", "3.0.0")

        def reformat_cnn(x):
            return {
                'question': f"Article: {x['article']}\nSummary:",
                'answers': {'text': [x['highlights']]},
                'context': x['article'],
                'id': str(x['id']),
            }

        train_dataset = [reformat_cnn(d) for d in dataset["train"]]
        validation_dataset = [reformat_cnn(d) for d in dataset["validation"]]

    elif dataset_name == "record":
        # Load the JSON file
        for split in ["train", "dev"]:
            dataset_dictionary = {
                "id": [], "question": [], "context": [], "answers": []}
            path = f"~/uncertainty/data/record/{split}.json"
            with open(path, "rb") as file:
                data = json.load(file)

            # Extract the relevant information and create a dictionary
            for item in data["data"]:
                for qa in item["qas"]:  # pylint: disable=invalid-name
                    dataset_dictionary["id"].append(qa["id"])
                    dataset_dictionary["question"].append(qa["query"])
                    dataset_dictionary["context"].append(
                        item["passage"]["text"])
                    list_of_answer_strings = []
                    list_of_answer_starts = []
                    for answer in qa["answers"]:
                        list_of_answer_strings.append(answer["text"])
                        list_of_answer_starts.append(answer["start"])
                    dataset_dictionary["answers"].append({
                        "text": list_of_answer_strings,
                        "answer_start": list_of_answer_starts})

            # Create the Hugging Face dataset
            if split == "train":
                train_dataset = datasets.Dataset.from_dict(dataset_dictionary)
                logging.info('train_dataset[0]: %s', train_dataset[0])
            else:
                validation_dataset = datasets.Dataset.from_dict(dataset_dictionary)
                logging.info('validation_dataset[0]: %s', validation_dataset[0])

    return train_dataset, validation_dataset
