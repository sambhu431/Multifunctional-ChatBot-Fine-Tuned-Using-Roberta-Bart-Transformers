---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:1681
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/stsb-roberta-base
widget:
- source_sentence: i received a bill for services i didnt receive what should i do
    unless i received a medical bill for sexual services i didnt dare receive what
    should i say do
  sentences:
  - a savings account is an interest-bearing deposit account held at a bank or other
    financial institution
  - yes we offer same-day appointments for urgent care needs availability may vary
    so its best to call early in the day to secure a slot
  - review the bill carefully and compare it with your medical records if the charge
    is incorrect contact our billing department with the details so we can investigate
    and resolve the issue
- source_sentence: i've actually been pretty good you that i ' ve actually still been
    pretty much good you
  sentences:
  - i'm actually in school right now
  - doctor usually calls you in  minutes right now due to high demand with an increase
    in covid cases doctors can take upto  hrs to call you
  - 'kindly ensure medicines in your order are eligible for cashless benefits please
    look for the below labels in medicine search while placing an order cashless:
    medicine is available under cashless benefits policy check required: our team
    will do an eligibility check  approve eligible medicines for cashless benefits
    cash on delivery / no label: medicine is not available under cashless benefits'
- source_sentence: what is are bjrnstad syndrome also what sometimes is mentioned
    are bjrnstad syndrome
  sentences:
  - 'although cashless hospitalization facility is available at the medi assist network
    of hospitals you may sometimes need to use hospitals that are not in the medi
    assist network reimbursement claims may be filed in the following circumstances:
    hospitalization at a non-network hospital post- and pre-hospitalization expenses
    / domiciliary expenses that are not covered by your policy denial of preauthorization
    for specific reasons'
  - bjrnstad syndrome is a rare disorder characterized by abnormal hair and hearing
    problems affected individuals have a condition known as pili torti which means
    "twisted hair" so named because the strands appear twisted when viewed under a
    microscope the hair is brittle and breaks easily leading to short hair that grows
    slowly in bjrnstad syndrome pili torti usually affects only the hair on the head;
    eyebrows eyelashes and hair on other parts of the body are normal the proportion
    of hairs affected and the severity of brittleness and breakage can vary this hair
    abnormality commonly begins before the age of  it may become milder with age particularly
    after puberty people with bjrnstad syndrome also have hearing problems that become
    evident in early childhood the hearing loss which is caused by changes in the
    inner ear sensorineural deafness can range from mild to severe mildly affected
    individuals may be unable to hear sounds at certain frequencies while severely
    affected individuals may not be able to hear at all
  - the minimum investment possible is rs  in sip and rs  in lump sum but at the very
    least rs  is suggested
- source_sentence: what is the recommended frequency for dental check-ups what is
    now the highest recommended frequency method for dental check - ups
  sentences:
  - good luck with that
  - every  months
  - hemophilia a and hemophilia b are inherited in an x-linked recessive pattern the
    genes associated with these conditions are located on the x chromosome which is
    one of the two sex chromosomes in males who have only one x chromosome one altered
    copy of the gene in each cell is sufficient to cause the condition in females
    who have two x chromosomes a mutation would have to occur in both copies of the
    gene to cause the disorder because it is unlikely that females will have two altered
    copies of this gene it is very rare for females to have hemophilia a characteristic
    of x-linked inheritance is that fathers cannot pass x-linked traits to their sons
    in x-linked recessive inheritance a female with one altered copy of the gene in
    each cell is called a carrier carrier females have about half the usual amount
    of coagulation factor viii or coagulation factor ix which is generally enough
    for normal blood clotting however about  percent of carrier females have less
    than half the normal amount of one of these coagulation factors; these individuals
    are at risk for abnormal bleeding particularly after an injury surgery or tooth
    extraction
- source_sentence: what are the genetic changes related to globozoospermia under what
    are then the genetic population changes related to cellular globozoospermia
  sentences:
  - this condition is inherited in an x-linked recessive pattern the gene associated
    with this condition is located on the x chromosome which is one of the two sex
    chromosomes in males who have only one x chromosome one altered copy of the gene
    in each cell is sufficient to cause the condition in females who have two x chromosomes
    a mutation would have to occur in both copies of the gene to cause the disorder
    because it is unlikely that females will have two altered copies of this gene
    males are affected by x-linked recessive disorders much more frequently than females
    a characteristic of x-linked inheritance is that fathers cannot pass x-linked
    traits to their sons
  - the  years rule is used to calculate the value of the fd to double divide the
    highest rate by  if the highest rate of the fd is  your fd will double with this
    basic formula / =  it will take less than  years for the fd to double
  - globozoospermia is most commonly caused by mutations in the dpyl gene which are
    found in about  percent of men with this condition mutations in other genes likely
    also cause globozoospermia the dpyl gene provides instructions for making a protein
    that is found in developing sperm cells the dpyl protein is involved in the development
    of the acrosome and elongation of the sperm head which are integral steps in sperm
    cell maturation mutations in the dpyl gene result in a loss of functional dpyl
    protein as a result sperm cells have no acrosome and do not elongate properly
    without an acrosome the abnormal sperm are unable to get through the outer membrane
    of an egg cell to fertilize it leading to infertility in affected men researchers
    have described other characteristics of the abnormal sperm cells that make fertilization
    of an egg cell difficult although it is not clear how changes in the dpyl gene
    are involved in development of these characteristics
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/stsb-roberta-base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/stsb-roberta-base](https://huggingface.co/sentence-transformers/stsb-roberta-base). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/stsb-roberta-base](https://huggingface.co/sentence-transformers/stsb-roberta-base) <!-- at revision 32d471df2968a46d1fe447d66a9275e8e63fcf12 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 128, 'do_lower_case': True}) with Transformer model: RobertaModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'what are the genetic changes related to globozoospermia under what are then the genetic population changes related to cellular globozoospermia',
    'globozoospermia is most commonly caused by mutations in the dpyl gene which are found in about  percent of men with this condition mutations in other genes likely also cause globozoospermia the dpyl gene provides instructions for making a protein that is found in developing sperm cells the dpyl protein is involved in the development of the acrosome and elongation of the sperm head which are integral steps in sperm cell maturation mutations in the dpyl gene result in a loss of functional dpyl protein as a result sperm cells have no acrosome and do not elongate properly without an acrosome the abnormal sperm are unable to get through the outer membrane of an egg cell to fertilize it leading to infertility in affected men researchers have described other characteristics of the abnormal sperm cells that make fertilization of an egg cell difficult although it is not clear how changes in the dpyl gene are involved in development of these characteristics',
    'this condition is inherited in an x-linked recessive pattern the gene associated with this condition is located on the x chromosome which is one of the two sex chromosomes in males who have only one x chromosome one altered copy of the gene in each cell is sufficient to cause the condition in females who have two x chromosomes a mutation would have to occur in both copies of the gene to cause the disorder because it is unlikely that females will have two altered copies of this gene males are affected by x-linked recessive disorders much more frequently than females a characteristic of x-linked inheritance is that fathers cannot pass x-linked traits to their sons',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 1,681 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                         |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                         |
  | details | <ul><li>min: 5 tokens</li><li>mean: 22.99 tokens</li><li>max: 69 tokens</li></ul> | <ul><li>min: 2 tokens</li><li>mean: 43.7 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                              | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>which exercise is beneficial for cardiovascular health which exercise act is thus beneficial for the cardiovascular health</code> | <code>running</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | <code>1.0</code> |
  | <code>how do i choose a mutual fund how do it i choose with a mutual guarantee fund</code>                                              | <code>selecting the right mutual fund as per your investment strategy and goal can be overwhelming here are some high level criteria to make your choice easier make a strategy: think about your financial situation goals timeline and risk tolerance before you select your investment with the help of these answers you will be able to make a choice based on size of company style credit quality etc with this you can make a balance in your portfolio monitor performance of the company: the performance of the stocks you are interested in should be monitored at all times periodically past performance alone cannot guarantee future performance but studying it over the long term may help you narrow down your choices think about the costs: you must look at the expense ratio of the funds you want to invest in as the costs are an important consideration your costs may include the management fee distribution fee transaction fee and other expenses fees differ from fund to fund so a look into it for compariso...</code> | <code>1.0</code> |
  | <code>what is are nonsyndromic paraganglioma but what then is rare are nonsyndromic acquired paraganglioma</code>                       | <code>paraganglioma is a type of noncancerous benign tumor that occurs in structures called paraganglia paraganglia are groups of cells that are found near nerve cell bunches called ganglia paragangliomas are usually found in the head neck or torso however a type of paraganglioma known as pheochromocytoma develops in the adrenal glands adrenal glands are located on top of each kidney and produce hormones in response to stress most people with paraganglioma develop only one tumor in their lifetime some people develop a paraganglioma or pheochromocytoma as part of a hereditary syndrome that may affect other organs and tissues in the body however the tumors often are not associated with any syndromes in which case the condition is called nonsyndromic paraganglioma or pheochromocytoma pheochromocytomas and some other paragangliomas are associated with ganglia of the sympathetic nervous system the sympathetic nervous system controls the "fight-or-flight" response a series of changes in the body d...</code> | <code>1.0</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 5
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 4.7170 | 500  | 0.3779        |


### Framework Versions
- Python: 3.11.11
- Sentence Transformers: 3.4.1
- Transformers: 4.48.3
- PyTorch: 2.5.1+cu124
- Accelerate: 1.3.0
- Datasets: 3.3.2
- Tokenizers: 0.21.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->