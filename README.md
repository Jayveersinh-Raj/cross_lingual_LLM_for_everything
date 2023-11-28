# cross_lingual_LLM_for_everything
This is prompt tuning project for Practical Machine learning and Deep Learning course at Innopolis University.
Jayveersinh Raj: j.raj@innopolis.university
Evgenii Evlampev: e.evlampev@innopolis.university

# Project description
A **Bloom-3b** 8 bit quantized model was fine tuned using PEFT and LoRA for 2 downstream tasks:
- Sentence correction
- Question Answering

# The deliverables
In the deliverables a presentation and final report is provided for further description.

# Training notebooks
The training notebooks can be replicated on a google colab with a 16 GB T4 GPU. 

# Data preparation
The data preparation for artifical dataset is within the respective directories.

# Checkpoints
[Sentence correction](https://huggingface.co/Jayveersinh-Raj/bloom-sentence-correction)

[Question answering](Jayveersinh-Raj/bloom-que-ans)

# How to use the checkpoints
## Sentence Correction

    import torch
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from IPython.display import display, Markdown

    peft_model_id = "Jayveersinh-Raj/bloom-sentence-correction"
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=False, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    qa_model = PeftModel.from_pretrained(model, peft_model_id)

    

    def make_inference(question):
      batch = tokenizer(f"### INCORRECT\n{question}\n\n### CORRECT\n", return_tensors='pt').to("cuda")

      with torch.cuda.amp.autocast():
        output_tokens = qa_model.generate(**batch, max_new_tokens=200)

      display(Markdown((tokenizer.decode(output_tokens[0], skip_special_tokens=True))))

      text = "I red a book last night"
      make_inference(text)

## Question Answering

    import torch
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    peft_model_id = "Jayveersinh-Raj/bloom-que-ans"
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=False, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Load the Lora model
    qa_model = PeftModel.from_pretrained(model, peft_model_id)

    from IPython.display import display, Markdown

    def make_inference(context, question):
      batch = tokenizer(f"### CONTEXT\n{context}\n\n### QUESTION\n{question}\n\n### ANSWER\n", return_tensors='pt').to("cuda")

      with torch.cuda.amp.autocast():
        output_tokens = qa_model.generate(**batch, max_new_tokens=200)

      display(Markdown((tokenizer.decode(output_tokens[0], skip_special_tokens=True))))

    context = ""
    question = "What is the best food?"
    
    make_inference(context, question)

# Data used for training
[Gujarati for sentence correction](https://huggingface.co/datasets/Jayveersinh-Raj/Gujarati-correct-incorrect-sent)

[Squadv2 for Question Answering](https://huggingface.co/datasets/squad_v2)

# Resulting notes
The model performs significantly even after downstreamed on 1000 samples for sentence correction, which is 0.02% of the entire training data for **Bloom-3b** large language model. Moreover, for question answering the model also works over multiple languages which were used in training the model. Hence, the prospect of cross lingual functionalities are vast with the following approach. However, due to low weight adapters, the swapping of adapters on base model according to task is optimal, and hence eliminates the baggage of finding datasets in multiple languages for a task, and training 100 different models for 100 different tasks. 
