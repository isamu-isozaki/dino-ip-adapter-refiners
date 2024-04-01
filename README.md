# dino-ip-adapter-refiners
Repository for training an IP adapter for DINO

 - [Wandb](https://wandb.ai/isamu/finetune-ldm-ip-adapter)
 - [Discussion Doc](https://docs.google.com/document/d/1t6Cr7iJdUXmsGn1Zfw2h-Dg30g-oanH4FRh8pzsaZnY/edit?tab=t.0#heading=h.d7ff3urp6olq)

## Installation
You can install the package using pip:

```bash
pip install git+https://github.com/isamu-isozaki/dino-ip-adapter-refiners.git
```

or you can use the `rye` package manager:

```bash
rye sync
```


## Training
### Download Photo Concept Dataset

Download the dataset from the following link:

[Photo Concept Dataset](https://huggingface.co/datasets/ptx0/photo-concept-bucket)

To download the images from urls, you can use the (img2dataset)[https://github.com/rom1504/img2dataset] library for efficient downloading.

```bash
 img2dataset --url_list photo-concept-bucket/ --input_format "parquet" \
    --url_col "url" --caption_col "cogvlm_caption" --output_format webdataset \
    --output_folder photo-concept-bucket-webdataset --processes_count 16 --thread_count 64 --image_size 1024 \
    --resize_only_if_bigger False --resize_mode="center_crop"  --skip_reencode True
```

### Pre-encoded Features

Create a .env file in the root of the project with the following variables:

```bash
PHOTO_CONCEPT=/path/to/photo-concept-bucket-webdataset
PHOTO_CONCEPT_PREENCODED=/save/path/
```

Because we're using webdatasets, you can save and load data from a remote location. We used
Google Cloud Storage for this purpose.

Run the following command to precompute the features for the dataset:

```bash
python precompute_data.py --dataset photo_concept --start_shard 0 --end_shard 56 --batch_size 32
```

Batch size can be adjusted to fit the memory of the machine, 32 works for 40GB of memory.

If adding text encoder embeddings and working with mosaic you can encode with

```bash
 python precompute_data.py --dataset="photo_concept" --start_shard=0 --end_shard=56 --batch_size=32 --encode_prompt --use_mosaic --compression=zstd
 ```

### Train Model

To train the model, you can use the following command:

```bash
.venv/bin/python src/dino_ip_adapter_refiners
```

If using Mosaic, you first need to download index.json from PHOTO_CONCEPT_PREENCODED to the cache file location you specified. It will be along the lines of
```bash
gsutil cp GCP_SAVE_PATH/index.json CACHE_FOLDER_LOCATION
```
