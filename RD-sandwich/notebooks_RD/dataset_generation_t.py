import os

import numpy as np

dtype = np.float32
# import tqdm
import os

import numpy
import torch
import torchvision
from huggingface_hub import (_CACHED_NO_EXIST, scan_cache_dir,
                             try_to_load_from_cache)
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          CLIPVisionModelWithProjection,
                          ViTForImageClassification)

filters = ["attn"]
model_names = ["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7b-hf"]
tensor_zoo_path = "/home/jgryu/Weight_compression/Wparam_dataset/model_param_tensor/"
save_path = "/home/jgryu/Weight_compression/Wparam_dataset/TFRecord/"

# # model_name = model_names[0]
# model_name = model_names[0]

import glob
import json
import random

import tensorflow as tf
from tqdm import tqdm


def check_contains_all_substrings(string, substrings):
    return all(substring in string for substring in substrings)


def check_contains_any_substrings(string, substrings):
    return any(substring in string for substring in substrings)


def serialize_example(slice):
    feature_dict = {"slice": tf.train.Feature(float_list=tf.train.FloatList(value=slice))}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()


for model_name in model_names:
    print(model_name)
    for d in [4096, 1024, 256, 128, 64, 32, 16]:
        print(f"dim = {d}")
        target_model_tensor_path = tensor_zoo_path + model_name
        tensor_path_list = glob.glob(f"{target_model_tensor_path}/**/*.npy", recursive=True)

        random.seed(100)

        # all_tensors = []
        tensors_train = []
        tensors_val = []

        for tensor_path in tqdm(tensor_path_list):
            if not check_contains_all_substrings(tensor_path, filters):
                continue

            t = np.load(tensor_path)
            # print(t.shape)
            t = t.astype(np.float32)
            t = t.reshape(-1, d)
            # print(t.shape)
            # all_tensors.append(t)

            indices = np.random.permutation(len(t))
            split_index = int(len(t) * 0.8)
            train_indices = indices[:split_index]
            val_indices = indices[split_index:]

            tensors_train.append(t[train_indices])
            tensors_val.append(t[val_indices])

        dataset = {}
        dataset["train"] = np.vstack(tensors_train)
        dataset["val"] = np.vstack(tensors_val)

        print(dataset["train"].shape, dataset["val"].shape)

        save_model_path = os.path.join(save_path, model_name.replace("/", "--"), "_".join(filters), f"d{d}")
        os.makedirs(save_model_path, exist_ok=True)

        mean_vector = dataset["train"].mean(axis=0)
        mean_value = dataset["train"].mean()
        std_vector = dataset["train"].std(axis=0)
        std_value = dataset["train"].std()
        # print(mean_value, std_value)

        filename = os.path.join(save_model_path, "_".join(filters) + f"_d{d}_train.tfrecord")
        np.save(filename.replace(".tfrecord", "_mean_vector.npy"), mean_vector)
        np.save(filename.replace(".tfrecord", "_std_vector.npy"), std_vector)
        np.save(filename.replace(".tfrecord", "_mean.npy"), mean_value)
        np.save(filename.replace(".tfrecord", "_std.npy"), std_value)

        for split in ["train", "val"]:
            print(f"## Saving {split} dataset ##")
            filename = os.path.join(save_model_path, "_".join(filters) + f"_d{d}_{split}.tfrecord")
            with tf.io.TFRecordWriter(filename) as writer:
                for slice in tqdm(dataset[split]):
                    example = serialize_example(slice)
                    writer.write(example)
