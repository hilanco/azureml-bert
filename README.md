# BERT modell tanítása Azure Machine Learning környezetben

A repozitórium a Microsoft ONNX Runtime BERT tanítási módszerét alkalmazza, Microsoft Azure Machine Learning platformon. Jelenlegi ismereteink szerint ez a leggyorsabb és legolcsóbb módszer saját BERT modell készítésére.

Az tanító szkriptek eredeti formájukban https://github.com/microsoft/onnxruntime-training-examples/tree/master/nvidia-bert itt érhetőek el. Az ONNX Runtime az NVIDIA PyTorch alapú módszerét használta fel a tanításhoz. A módszer tartalmazza a jelenleg leggyorsabb Microsoft DeepSpeed optimizációkat is.

Az eredeti módszer használható Azure Machine Learning (AzureML) platformon, de futtatható lokálisan is. A megoldás továbbá használható tetszőleges méretű BERT modell tanítására, akár Tiny akár Large modell tanítására is.

A lentiek leírás részben az eredeti dokumentációra támaszkodik, azt kiegészítve hasznos tippekkel és tapasztalatokkal a tanítás során. A leírásban továbbá megtalálható a Wikipedia adathalmaz feldolgozása is, ami kiválóan használható Base vagy kisebb modellek tanítására is.

A tanításhoz további alábbi cikket (https://towardsdatascience.com/train-bert-large-in-your-own-language-7685ee26b05b) ajánljuk.

## Beállítások

A megoldás beállításához két repozitóriumot kell leklónozni. Ehhez AzureML platformon érdemes kiválasztani egy kisebb virtuális gépet (VM) erre a STANDARD_DS1_V2 tökéletes lehet.

Tehát akkor a kód klónozása két részletben:

1. ONNX Runtime 

    ```bash
    git clone https://github.com/microsoft/onnxruntime-training-examples.git
    cd onnxruntime-training-examples
    ```

2. NVIDIA BERT

    ```bash
    git clone --no-checkout https://github.com/NVIDIA/DeepLearningExamples.git
    cd DeepLearningExamples/
    git checkout 4733603577080dbd1bdcd51864f31e45d5196704
    cd ..
    ```

3. A kettő összehangolása + wikiextraktor letöltése

    ```bash
    mkdir -p workspace
    mv DeepLearningExamples/PyTorch/LanguageModeling/BERT/ workspace
    rm -rf DeepLearningExamples
    cp -r ./nvidia-bert/ort_addon/* workspace/BERT
    cd workspace
    pip3 install wikiextractor
    ```

4. Formázáshoz szükséges (saját) repozitórium klónozása

    ```bash
    git clone https://github.com/roberthajdu92/nvidia-bert-wiki-preprocess
    ```
    

## Adathalmaz előkészítése

Az adathalmaz előkészítéséhez érdemes már egy kicsit nagyobb VM-et választani, jó opciók lehetnek a Standard_D14_v2 vagy a Standard_D13_v2. Az adathalmaz alapvetően két követelménynek kell megfeleljen:

* Minden mondat egy sorban van.
* Minden bekezdést egy üres sor választ el.

1. Követelmények:

    * Python 3.6
    * Quntoken ```pip3 install quntoken```

2. Wikicorpus letöltése és előkészítése bináris formátummá:

    ### wikicorpus_hu letöltése és előkészítése
    ```bash
    mkdir -p ./workspace/BERT/data/download/wikicorpus_hu
    cd ./workspace/BERT/data/download/wikicorpus_hu
    wget https://dumps.wikimedia.org/huwiki/20210320/huwiki-20210320-pages-articles-multistream.xml.bz2
    bzip2 -dv huwiki-20210320-pages-articles-multistream.xml.bz2
    mv huwiki-20210320-pages-articles-multistream.xml wikicorpus_hu.xml
    cd ../../../../..
    ```

    ### wikicorpus formázása wikiextraktorral

    ```bash
    python3 -m wikiextractor.WikiExtractor wikicorpus_hu.xml
    ```

    ### wikicorpus további formázás, a klónozott szkriptekkel

    ```bash
    python3 formatting.py --input_folder=INPUT_FOLDER --output_file=OUTPUT_FILE
    ```

    ### wikicorpus tokenizálása quntokennel

    ```bash
    quntoken -f spl -m sentence < INPUT_FILE > OUTPUT_FILE
    ```

    ### wikicorpus filterelése a TurkuNLP csoport egy megoldásával

    ```bash
    python filtering.py INPUT_SPL_FILE \
    --word-chars abcdefghijklmnopqrstuvwxyzáéíóöúüőű \
    --language hu \
    --langdetect hu \
    > OUTPUT_FILE
    ```

    ### WordPiece vocab file készítése a kinyert szövegből

    ```bash
    python3 spmtrain.py INPUT_FILE \
    --model_prefix=bert \
    --vocab_size=32000 \
    --input_sentence_size=100000000 \
    --shuffle_input_sentence=true \
    --character_coverage=0.9999 \
    --model_type=bpe
    ```

    utána pedig

    ```bash
    python3 sent2wordpiece.py bert.vocab > vocab.txt
    ```

    ### Ezután a corpus darabolása következik

    ```bash
    python3 sharding.py \
    --input_file=INPUT_FILE \
    --num_shards=50
    ```
    
    ### Ezután a bináris fájlok elkészítése következik

    ```bash
    python3 create_hdf5_files.py --max_seq_length 128 --max_predictions_per_seq 20 --vocab_file=vocab.txt --n_processes=15
    ```

    illetve

    ```bash
    python3 create_hdf5_files.py --max_seq_length 512 --max_predictions_per_seq 80 --vocab_file=vocab.txt --n_processes=15
    ```

    ekkor a végén két könyvtárat fogunk kapni a bináris fileokkal:

    ```bash
    hdf5_lower_case_0_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5
    ```

    ```bash
    hdf5_lower_case_0_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5
    ```

## BERT pre-training with ONNX Runtime in Azure Machine Learning

1. Data Transfer

    * Transfer training data to Azure blob storage

    To transfer the data to an Azure blob storage using [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest), use command:
    ```bash
    az storage blob upload-batch --account-name <storage-name> -d <container-name> -s ./workspace/BERT/data
    ```

    * Register the blob container as a data store
    * Mount the data store in the compute targets used for training

    Please refer to the [storage guidance](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data#storage-guidance) for details on using Azure storage account for training in Azure Machine Learning. 

2. Execute pre-training

    The BERT pre-training job in Azure Machine Learning can be launched using either of these environments:

    * Azure Machine Learning [Compute Instance](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-instance) to run the Jupyter notebook.
    * Azure Machine Learning [SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py)

    You will need a [GPU optimized compute target](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets#amlcompute) - _either NCv3 or NDv2 series_, to execute this pre-training job.

    Execute the steps in the Python notebook [azureml-notebooks/run-pretraining.ipynb](azureml-notebooks/run-pretraining.ipynb) within your environment. If you have a local setup to run an Azure ML notebook, you could run the steps in the notebook in that environment. Otherwise, a compute instance in Azure Machine Learning could be created and used to run the steps.

## BERT pre-training with ONNX Runtime directly on ND40rs_v2 (or similar NVIDIA capable Azure VM) 

1. Check pre-requisites

    * CUDA 10.2
    * Docker
    * [NVIDIA docker toolkit](https://github.com/NVIDIA/nvidia-docker)

2. Build the ONNX Runtime Docker image

    Build the onnxruntime wheel from source into a Docker image.
    ```bash
    cd nvidia-bert/docker
    bash build.sh
    cd ../..
    ```    
    - Tag this image __onnxruntime-pytorch-for-bert__`
    
    To build and install the onnxruntime wheel on the host machine, follow steps [here](https://github.com/microsoft/onnxruntime/blob/master/BUILD.md#Training)

3. Set correct paths to training data for docker image.

   Edit `nvidia-bert/docker/launch.sh`.

   ```bash
   ...
   -v <replace-with-path-to-phase1-hdf5-training-data>:/data/128
   -v <replace-with-path-to-phase2-hdf5-training-data>:/data/512
   ...
   ```

   The two directories must contain the hdf5 training files.

4. Set the number of GPUs and per GPU limit.

    Edit `workspace/BERT/scripts/run_pretraining_ort.sh`.

    ```bash
    num_gpus=${4:-8}
    gpu_memory_limit_gb=${26:-"32"}
    ```

5. Modify other training parameters as needed.

    Edit `workspace/BERT/scripts/run_pretraining_ort.sh`.

    ```bash
    seed=${12:-42}

    accumulate_gradients=${10:-"true"}
    deepspeed_zero_stage=${27:-"false"}

    train_batch_size=${1:-16320}
    learning_rate=${2:-"6e-3"}
    warmup_proportion=${5:-"0.2843"}
    train_steps=${6:-7038}
    accumulate_gradients=${10:-"true"}
    gradient_accumulation_steps=${11:-340}

    train_batch_size_phase2=${17:-8160}
    learning_rate_phase2=${18:-"4e-3"}
    warmup_proportion_phase2=${19:-"0.128"}
    train_steps_phase2=${20:-1563}
    gradient_accumulation_steps_phase2=${11:-1020}
    ```
    The above defaults are tuned for an Azure NC24rs_v3.

    The training batch size refers to the number of samples a single GPU sees before weights are updated. The training is performed over _local_ and _global_ steps. A local step refers to a single backpropagation execution on the model to calculate its gradient. These gradients are accumulated every local step until weights are updated in a global step. The _microbatch_ size is samples a single GPU sees in a single backpropagation execution step. The microbatch size will be the training batch size divided by gradient accumulation steps.
    
    Note: The effective batch size will be (number of GPUs) x train_batch_size (per GPU). In general we recommend setting the effective batch size to ~64,000 for phase 1 and ~32,000 for phase 2. The number of gradient accumulation steps should be minimized without overflowing the GPU memory (i.e. maximizes microbatch size).

    Consult [Parameters](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#parameters) section by NVIDIA for additional details.

6. Launch interactive container.

    ```bash
    cd workspace/BERT
    bash ../../nvidia-bert/docker/launch.sh
    ```

7. Launch pre-training run

    ```bash
    bash /workspace/bert/scripts/run_pretraining_ort.sh
    ```

    If you get memory errors, try reducing the batch size or enabling the partition optimizer flag.

## Fine-tuning

For fine-tuning tasks, follow [model_evaluation.md](model_evaluation.md)
