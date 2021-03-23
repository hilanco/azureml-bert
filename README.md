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

    ### Utolsó lépésként pedig a két mappát BLOB tárolóba kell átmozgatni

    [azcopy](https://docs.microsoft.com/hu-hu/azure/storage/common/storage-use-azcopy-v10) paranccsal:

    ```bash
    azcopy.exe cp --recursive "src" "dest"
    ```

3. Tanítás futtatása AzureML platformon

Miután elkészítettük a wikicorpus adathalmazt és átmozgattuk a BLOB tárolóba, a tanító szkript, notebook átnézése következik. Ebben a repozitóriumban elkészítettük egy, magyar nyelvű verzióját a tanító notebooknak, ami segítséget nyújhat a tanítás, illetve a tanítás folyamatának megértése során. Ezt a notebookot akár be lehet másolni az eredeti helyére és ezt használni, illetve az eredeti, angol verziót is lehet használni.

További információk tehát a repozitóriumban található [Notebook](https://github.com/roberthajdu92/azureml-bert/blob/master/run-pretraining.ipynb)-ban.