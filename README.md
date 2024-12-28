# NLP-Authorship-Attribution
#
# src saved datasets mapping:
#       domain_dataset - loaded base dataset from hugging face with human and gpt articles
#       domain_dataset_generated - includes llama and falcon article generation
#       domain_dataset_output - held out semi dataset using base models (non instruct models)
#       domain_dataset_clean - final result after post proccessing, removal of instruction artifacts