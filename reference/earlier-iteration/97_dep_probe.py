# Databricks notebook source
print("env ok")
import importlib.metadata as md
for p in ["diffusers","transformers","huggingface_hub","bitsandbytes","mosaicml-streaming","accelerate","pillow"]:
    try: print(p, md.version(p))
    except Exception as e: print(p, "absent")
