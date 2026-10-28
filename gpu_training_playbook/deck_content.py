"""Deck content: the single source of truth for BOTH output formats.

build_deck.py renders this to Google Slides; build_html.py renders it to a
self-contained HTML deck. Edit slides HERE so the two can never drift.

Each entry: (layout_key, title, body_lines | None, table_rows | None)
layout_key is a symbolic name; build_deck.py maps it to a template layout id.
"""

L_TITLE = "L_TITLE"
L_CONTENT = "L_CONTENT"
L_TITLE_ONLY = "L_TITLE_ONLY"
L_DIVIDER = "L_DIVIDER"

SLIDES = [
    (L_TITLE, "Training DL & LLMs on Databricks GPUs",
     ["Setup, techniques, templates, and the unvarnished truths"], None),

    (L_CONTENT, "Agenda",
     ["The two GPU paths, and how to choose",
      "GPU training concepts: a shared vocabulary",
      "The training lifecycle on Databricks",
      "Setup walkthrough: which knobs, and which to leave alone",
      "Techniques: data loading, checkpointing, tracking, registration",
      "What changed: packages no longer recommended",
      "Templates you can start from today",
      "Measured results",
      "The unvarnished truths",
      "Getting the model out"], None),

    (L_TITLE_ONLY, "Two paths to a GPU", None,
     [["", "AI Runtime (serverless)", "Classic (Runtime ML)"],
      ["What you manage", "Nothing: pick an accelerator", "Nodes, runtime, scaling, termination"],
      ["Compute runs in", "Databricks-managed plane", "Your own cloud account"],
      ["Time to first GPU", "~40 seconds (measured)", "~10 minutes (measured)"],
      ["GPUs", "A10, H100, B300", "T4, L4, A10G, L40S, V100, A100, H100"],
      ["Spark in same job", "No (Spark Connect only)", "Yes, full cluster"],
      ["Status", "GA Oct 15, 2026", "Long-established, LTS"]]),

    (L_CONTENT, "Default to AI Runtime",
     ["Databricks' own guidance, quoted: for GPU deep learning, fine-tuning and batch inference, "
      "\"Databricks recommends AI Runtime ... instead of Runtime ML\"",
      "Cited reasons: lower total cost of ownership, a more flexible developer experience, multinode scale",
      "Classic remains the right answer in specific cases: account separation, Spark in-cluster, "
      "Ray-on-Spark, GPU types AI Runtime lacks, or an unsupported compliance profile",
      "See the appendix for the full decision list"], None),

    (L_CONTENT, "GPU concepts: the vocabulary",
     ["Single-node, single-GPU: where every project should start",
      "Single-node, multi-GPU: one machine, several GPUs, data-parallel",
      "Multi-node: several machines, GPUs synchronizing over the network",
      "Data parallelism (DDP): replicate the model on each GPU, each reads a different data shard, "
      "compute gradients, average gradients across all GPUs, update, repeat",
      "The concepts are durable. Only the tooling churns."], None),

    (L_CONTENT, "Why the interconnect is the ceiling",
     ["Gradient averaging is a collective operation (ring all-reduce) on every single step",
      "Each GPU sends gradients to a neighbour and receives from another, around a logical ring",
      "So at scale the network, not the GPU, becomes the bottleneck",
      "This is why multi-node training is sensitive to the fabric, and why RDMA-class interconnect "
      "is a real differentiator rather than a marketing bullet",
      "It is also why you climb the scaling ladder in order instead of jumping to multi-node"], None),

    (L_CONTENT, "The scaling ladder",
     ["1. Single node, single GPU: establish a correct, reproducible baseline",
      "2. Fix the data pipeline if it is the bottleneck. It often is (see our measured results)",
      "3. Single node, multi-GPU",
      "4. Multi-node",
      "5. Hyperparameter search, only once the baseline is stable",
      "Distributed code is harder to debug and can be slower from communication overhead. "
      "Distribute when the model or data will not fit, not by default."], None),

    (L_CONTENT, "The training lifecycle on Databricks",
     ["Data (UC Volumes / Delta) -> Train (GPU compute) -> Track (MLflow) -> "
      "Register (Unity Catalog) -> Deploy (Model Serving or export off-platform)",
      "Governance runs through the entire chain: the data, the experiment, the model, and the "
      "lineage between them all live in Unity Catalog",
      "That is the part a bare GPU cluster somewhere else does not give you",
      "The deck is organized along this lifecycle; the runtime is a swappable parameter"], None),

    (L_CONTENT, "Setup: AI Runtime, three ways in",
     ["Notebook: environment panel -> pick accelerator -> Apply. Attaches in a couple of minutes",
      "CLI: databricks air run --file workload.yaml",
      "Jobs and DABs: ai_runtime_task (not a notebook task)",
      "The YAML is small. experiment_name, environment.version, compute.accelerator_type, "
      "code_source, command. That is the whole surface.",
      "Tip: databricks air run -h config.<field> is a built-in schema browser, and it is more "
      "current than the docs page"], None),

    (L_TITLE_ONLY, "AI Runtime: the knobs that matter", None,
     [["Knob", "Guidance"],
      ["accelerator_type", "1xA10 and 1xH100 are SINGLE-GPU only. Multi-GPU needs 8xH100 / 8xB300"],
      ["environment.version", "'databricks_ai_v6' = AI image with torch. Plain 6 = Standard image, NO torch"],
      ["num_accelerators", "Defaults to all GPUs on the node"],
      ["timeout_minutes", "Default is 3 hours. Set it, or a hung job bills for hours"],
      ["max_retries", "Defaults to RETRYING, including deterministic failures. Set 0 while iterating"],
      ["dependencies", "Installed with uv, 15 GB cap. Notebooks cache; CLI and jobs reinstall every run"],
      ["secrets", "Use for HF_TOKEN. Hub egress works but is rate-limited unauthenticated"]]),

    (L_TITLE_ONLY, "Classic GPU: the knobs that matter", None,
     [["Knob", "Set to", "Why"],
      ["Policy", "Unrestricted", "The default 'Personal Compute' button allows ONE GPU type (g4dn.xlarge, T4)"],
      ["Machine learning", "ON", "Gives DBR ML with torch/CUDA, and forces Dedicated mode which UC needs"],
      ["Runtime", "18.3 LTS ML (GPU)", "LTS. Picking a GPU node filters the runtime list"],
      ["Node type", "g5.xlarge (A10G 24GB)", "g6e for more VRAM; p5 (H100 80GB) for large LLMs"],
      ["Auto-terminate", "20-30 min", "THE most important cost knob. Defaults are 120 min, or 3 days on Personal"],
      ["Photon", "OFF", "Not supported with GPU instance types at all"],
      ["Spot", "OFF", "Spot GPUs are documented as difficult to retain"]]),

    (L_CONTENT, "Job compute vs all-purpose compute",
     ["Job compute for recorded runs: cheaper DBUs, terminates automatically, spec lives in code",
      "All-purpose only for the interactive debug loop, where a warm cluster turns a 6-minute "
      "iteration into seconds",
      "Methodological point: AI Runtime is ephemeral per-run compute. Job compute is the honest "
      "comparison, because it also pays its start cost every run",
      "Timing classic on a pre-warmed all-purpose cluster hides the 10-minute spin-up",
      "NCCL escapes: NCCL_P2P_DISABLE=1 for peer-access errors on G5; "
      "NCCL_SOCKET_IFNAME=eth0 for multi-node NCCL failures"], None),

    (L_TITLE_ONLY, "Techniques: getting data to the GPU", None,
     [["Source", "AI Runtime", "Classic"],
      ["UC Volumes (files)", "Filesystem paths; UCVolumeDataset + DataLoader", "Filesystem paths (Dedicated mode)"],
      ["Delta / tables", "Spark Connect -> toPandas; or write parquet to a Volume", "Spark DataFrame directly"],
      ["Large cloud datasets", "(not documented for AIR)", "Mosaic Streaming, MDS format"],
      ["S3 direct", "Not covered in docs", "Not covered in docs"]]),

    (L_CONTENT, "Two things to know about the data path",
     ["Mosaic Streaming appears NOWHERE in the AI Runtime docs. The runtimes genuinely have "
      "different recommended data paths. This is not the same script with a different compute target.",
      "File sizing matters: aim for 1 to 10 GiB files. For many small files, copy in parallel first.",
      "Our own measured ViT gap turned out to be a data-pipeline effect, not a GPU effect",
      "Checkpointing on AI Runtime: PyTorch Distributed Checkpoint, dcp.save / dcp.load with "
      "UCVolumeWriter and UCVolumeReader. NOT torch.save.",
      "Always checkpoint. There is no capacity SLA.",
      "Under /Volumes/{cat}/{schema}/, every first-level directory must be a registered volume or "
      "os.makedirs fails with OSError errno 95"], None),

    (L_CONTENT, "Techniques: tracking and registration",
     ["Set the experiment explicitly. AI Runtime derives one from experiment_name; classic .py "
      "tasks get none at all and fail with 'Could not find experiment with ID None'",
      "HuggingFace Trainer: report_to=['mlflow']. Lightning: mlflow.pytorch.autolog()",
      "AI Runtime auto-logs GPU/CPU/memory system metrics and run logs (60-day retention). "
      "Classic does not do this automatically.",
      "Log metrics from rank 0 only in distributed runs",
      "Register to UC: mlflow.set_registry_uri('databricks-uc'), name as catalog.schema.model. "
      "A signature is required."], None),

    (L_CONTENT, "The model-logging trap",
     ["mlflow.pytorch.log_model fails on an Accelerate/AMP-prepared model: "
      "PicklingError: Cannot pickle a prepared model with automatic mixed precision",
      "And bf16 weights break signature inference: Input type (float) and bias type (BFloat16) "
      "should be the same",
      "Fix, three lines:",
      "   m = trainer.accelerator.unwrap_model(trainer.model)",
      "   m = m.to('cpu').to(torch.float32).eval()",
      "   mlflow.transformers.log_model(transformers_model={'model': m, 'tokenizer': tok}, ...)",
      "For LoRA, log the MERGED fp32 model, not the adapter, if it must load without PEFT"], None),

    (L_TITLE_ONLY, "What changed: stop using these", None,
     [["Was", "Now", "Status of the old thing"],
      ["HorovodRunner / Horovod", "TorchDistributor, @distributed", "Deprecated; not preinstalled after 15.4 LTS"],
      ["Petastorm", "Mosaic Streaming (MDS)", "Deprecated, explicitly"],
      ["HyperOpt", "Optuna, or Ray Tune", "Unmaintained; removed from DBR ML after 16.4 LTS"],
      ["Foundation Model Fine-tuning", "AI Runtime", "END OF LIFE (docs still link it as current)"],
      ["serverless_gpu package", "databricks.air", "Deprecated (FutureWarning)"],
      ["P3 / V100 instances", "G5, G6, G6e, P4, P5", "Deprecated"],
      ["TensorFlow as top-tier", "PyTorch", "Not top-tier from DBR 18.0 ML"],
      ["Workspace model registry", "UC registry + aliases", "Legacy"]]),

    (L_CONTENT, "Pin versions; distrust some docs",
     ["The two runtimes ship a DIFFERENT MAJOR VERSION of transformers: 5.8.1 on AI Runtime v6 "
      "versus 4.57.1 on DBR 18 ML",
      "Also torch 2.11 / CUDA 13.0 versus torch 2.9 / CUDA 12.9",
      "Code is not guaranteed to move between them unpinned. Pin explicitly rather than relying "
      "on preinstalled defaults.",
      "Good news: peft and trl pinned to the same versions installed cleanly on both",
      "Stale docs to expect: the public compute/gpu page still claims CUDA 11.0 while DBR 17.3+ "
      "ships 12.6+; train-model/pytorch pins torch==1.5.0 in its example; dl-best-practices never "
      "mentions AI Runtime at all"], None),

    (L_CONTENT, "Unvarnished truths: AI Runtime (1 of 2)",
     ["No capacity SLA and no startup SLA. On a capacity miss the session sits in 'starting' "
      "then fails, with no durable queue. Checkpoint and retry with backoff.",
      "Cross-region bursting is real and not pinnable. The Geo setting enforces Geo, not region.",
      "No customer-configurable GPU or spend caps, and no reserved capacity today.",
      "No self-service off switch post-GA.",
      "Access control is coarse: serverless ACLs are per-object (Interactive / Automated), "
      "NOT per-GPU. You cannot express 'GPU training only'."], None),

    (L_CONTENT, "Unvarnished truths: AI Runtime (2 of 2)",
     ["Notebook @distributed is single-NODE. Multi-node requires the CLI plus torchrun.",
      "1xA10 and 1xH100 are single-GPU. Multi-GPU needs 8xH100 or 8xB300.",
      "1xH100 is not a valid serverless GPU JOB config. Use 1xA10 or 8xH100.",
      "for_each is unsupported, and dynamic {{}} references do not work in ai_runtime_task.",
      "Default quota: 8 nodes x 8xH100 = 64 GPUs per workspace.",
      "Compute runs in the Databricks-managed plane. This cannot satisfy a cloud-account-"
      "separation requirement; no customer-hosted variant exists."], None),

    (L_CONTENT, "Unvarnished truths: classic GPU",
     ["About 10 minutes to a usable GPU, every single run",
      "Idle clusters bill. Auto-terminate is the only guard, and the defaults are generous.",
      "Spot GPUs are hard to retain; quota increases are often needed",
      "Photon does not apply to GPU instance types",
      "No automatic system-metrics capture into MLflow",
      "The honest upside: compute runs in your own cloud account, and the runtime is genuinely "
      "batteries-included. Our ViT run needed ZERO added libraries."], None),

    (L_TITLE_ONLY, "Performance tuning: where the wins are", None,
     [["Tier", "Knob", "Note"],
      ["1. Data pipeline", "dataloader_num_workers", "Match to vCPU. We ran 4 on a 4-vCPU node: likely contending"],
      ["1. Data pipeline", "pin_memory, persistent_workers, prefetch_factor", "Cheap, usually free throughput"],
      ["1. Data pipeline", "Pre-resize / pre-tokenize offline", "Usually the single biggest win for image work"],
      ["1. Data pipeline", "File layout", "Aim for 1-10 GiB files; many small files hurt badly"],
      ["2. GPU memory", "per_device_train_batch_size", "Our LoRA peaked at 4.99 of 23 GB: room for ~4x"],
      ["2. GPU compute", "bf16, torch.compile, TF32", "bf16 already on in our runs"],
      ["2. GPU compute", "Flash Attention 2", "flash_attn is preinstalled on DBR ML"],
      ["2. GPU memory", "gradient_checkpointing", "Saves memory, costs 20-30% speed. Only when memory-bound"],
      ["3. LoRA", "lora_r / alpha / target_modules", "We used 16/32 and all 7 modules = 1.75% trainable"],
      ["4. Platform", "Accelerator, multi-GPU, multi-node", "A10 -> H100; 8xH100 + @distributed; FSDP / ZeRO-3"]]),

    (L_CONTENT, "Tune where the evidence points",
     ["Our own measurements located the bottleneck: the ViT gap between runtimes was 26.9%, but "
      "the LoRA gap was only 2.7%. Same GPU both times.",
      "That difference is the data pipeline, not the GPU. ViT decodes and preprocesses JPEGs on the "
      "host every batch; LoRA feeds pre-tokenized text and is GPU-matmul-bound.",
      "So for image and video work, profile the input pipeline BEFORE buying a bigger GPU.",
      "The most obviously untapped knob in our own runs: LoRA peaked at 4.99 GB on a 23 GB card. "
      "Batch size could rise roughly 4x before OOM. That is free throughput we left on the table.",
      "Measure, change one knob, measure again. Single cold runs mislead."], None),

    (L_TITLE_ONLY, "Multimodal coverage and the one real gap", None,
     [["Modality", "Reference example", "Where"],
      ["Vision-language (VLM)", "LoRA on Nemotron 3.5 Super VL 120B-A12B, multi-GPU", "Cookbook post-training"],
      ["Vision backbone", "Image classification fine-tune, DINOv3", "Cookbook post-training"],
      ["Video", "Video action classification", "Cookbook training"],
      ["Audio / speech", "Whisper large-v3-turbo batch inference", "Docs tutorial"],
      ["Object detection", "YOLO11n on COCO128", "Docs tutorial"],
      ["Embeddings", "Qwen3 text embeddings for search", "Cookbook batch-inference"],
      ["DIFFUSION (SD / SDXL / Flux)", "NONE. No reference implementation exists.", "Net-new work"]]),

    (L_CONTENT, "Notes on the multimodal examples",
     ["The VLM example is real but large: Nemotron Super VL at 120B needs 8xH100. It will not fit "
      "on an A10. Plan the accelerator before promising a demo.",
      "Diffusion is the genuine gap. There is no Stable Diffusion, SDXL, Flux, DreamBooth, LoRA-on-"
      "SDXL or ControlNet example anywhere in the cookbook or the docs tutorials.",
      "diffusers is also not in the preinstalled stack, so it would need declaring as a dependency.",
      "If asked 'can we fine-tune an image generator here': yes technically, but there is no "
      "reference implementation to point at, and that should be said plainly.",
      "Multimodal data loading genuinely differs from both pure-vision and pure-text: the processor "
      "handles interleaved image and text, so collation and sequence budgeting change."], None),

    (L_CONTENT, "Templates to start from today",
     ["Official tutorials under machine-learning/ai-runtime/examples/tutorials/:",
      "   sgc-finetune-qwen3.5-0.8b (end-to-end fine-tune and serve), sgc-finetune-qwen3-4b, "
      "sgc-rl-gemma4-2b (GRPO), sgc-yolo11n-detect-coco128 (CV), sgc-ray-resnet18, sgc-tabfm",
      "Cookbook: github.com/databricks-solutions/ai-runtime-cookbook",
      "This playbook, parameterized and runnable on both runtimes:",
      "   00_setup.py: stage a dataset to a UC Volume as parquet, persist label names",
      "   01_finetune_vision.py: ViT image classification",
      "   02_finetune_llm_lora.py: LoRA SFT, merges and registers a portable fp32 model",
      "Both accept env vars AND --KEY=value args, because AI Runtime passes env_variables in YAML "
      "while classic spark_python_task takes parameters"], None),

    (L_TITLE_ONLY, "Measured: ViT-B/16 fine-tune", None,
     [["", "AI Runtime", "Classic DBR 18 ML"],
      ["Compute start", "~40 s", "621 s"],
      ["Training loop", "48.4 s", "38.2 s"],
      ["Throughput", "64.1 img/s", "81.3 img/s"],
      ["Eval accuracy", "0.985", "1.000"],
      ["TOTAL WALL", "183 s", "852 s"]]),

    (L_TITLE_ONLY, "Measured: LoRA SFT, Qwen2.5-0.5B", None,
     [["", "AI Runtime", "Classic (warm)"],
      ["Training loop", "192.1 s", "187.1 s"],
      ["Throughput", "10.41 samp/s", "10.69 samp/s"],
      ["Train / eval loss", "2.1520 / 2.127", "2.1519 / 2.126"],
      ["Trainable params", "1.75% (8.8M of 502.8M)", "identical"],
      ["Peak GPU memory", "4.99 GB", "6.26 GB"]]),

    (L_TITLE_ONLY, "Measured: VLM LoRA, Qwen2-VL-2B", None,
     [["", "AI Runtime", "Classic DBR 18 ML"],
      ["transformers version", "5.8.1", "4.57.1"],
      ["Compute start", "~40 s", "621 s"],
      ["Training loop", "38.3 s", "47.6 s"],
      ["Throughput", "6.259 samp/s", "5.041 samp/s"],
      ["Train / eval loss", "2.5580 / 1.7946", "2.5483 / 1.7865"],
      ["Trainable params", "0.197% (4.4M of 2.21B)", "identical"],
      ["Peak GPU memory", "6.34 GB", "6.34 GB (identical)"],
      ["TOTAL WALL", "152 s", "869 s"]]),

    (L_TITLE_ONLY, "Per-step: no consistent winner", None,
     [["Workload", "Nature", "Faster per step"],
      ["ViT fine-tune", "Host-heavy JPEG decode + preprocessing", "Classic +26.9%"],
      ["LoRA SFT (text)", "GPU-matmul-bound, pre-tokenized", "Classic +2.7% (noise)"],
      ["VLM LoRA", "Image + text through one processor", "AI RUNTIME +24.2%"]]),

    (L_CONTENT, "What that table actually means",
     ["The direction FLIPS by workload. So 'classic compute is faster per step' is not a true "
      "statement, and neither is the reverse.",
      "The most likely driver is the library version gap, not the platform: AI Runtime ships "
      "transformers 5.8.1 and torch 2.11; classic ships 4.57.1 and torch 2.9. Qwen2-VL support "
      "improved substantially across that major version.",
      "So AI Runtime wins where the newer stack helps, and loses where host-side data loading "
      "dominates.",
      "Honest statement: per-step throughput depends on YOUR workload and YOUR library versions, "
      "not on serverless versus classic. Pin versions and measure your own workload.",
      "End-to-end, serverless still wins on anything short, because of the 621-second provisioning."], None),

    (L_CONTENT, "Multimodal data loading is different",
     ["Confirmed by building it. A VLM sample needs:",
      "   1. Chat-template messages carrying an image placeholder",
      "   2. Text and images through ONE processor call together",
      "   3. A custom collator. The stock text collator cannot do this.",
      "   4. Masking image-placeholder and pad tokens out of the loss",
      "   5. Sequence budgeting for image tokens, which dominate the text (max_pixels)",
      "   6. remove_unused_columns=False, or Trainer strips the image column before collation",
      "Freezing the vision tower and adapting only the language projections gave 0.197% trainable, "
      "an order of magnitude below the text-only LoRA's 1.75%.",
      "Portability: the VLM model class moved between transformers majors. The template tries "
      "Qwen2VLForConditionalGeneration and falls back to AutoModelForVision2Seq."], None),

    (L_CONTENT, "Read three things off those tables",
     ["1. Serverless is 4.7x to 5.7x faster END TO END on short work, almost entirely because of "
      "cluster provisioning (621 s, measured identically on two separate job-compute runs).",
      "2. Per-step throughput has no consistent winner. It flips by workload and tracks the "
      "library versions, not the platform. Measure your own workload.",
      "3. Training is equivalent across runtimes: losses match to 3-4 decimal places, peak memory "
      "matched exactly on the VLM, and the VLM produced identical generated text, all across "
      "different transformers major versions.",
      "Headroom: a 0.5B text model peaked under 5 GB, and a 2.2B VLM under 6.4 GB, on a 24 GB A10. "
      "You do not need H100s to start.",
      "Caveat stated plainly: single cold runs, n=1 per cell. Directional, not benchmarks."], None),

    (L_CONTENT, "Getting the model out",
     ["Relevant whenever inference must run somewhere other than Databricks: on-prem, at the "
      "edge, or in your own cloud account.",
      "The honest framing: this path is THINLY DOCUMENTED. The model-lifecycle page presents "
      "download_artifacts as an advanced/debug use case, and the Model Serving docs contain no "
      "export or on-prem guidance at all.",
      "What works today:",
      "   mlflow.artifacts.download_artifacts(",
      "       artifact_uri='models:/cat.schema.model/1', dst_path='/Volumes/.../export')",
      "Use /Volumes destinations. /dbfs paths fail."], None),

    (L_CONTENT, "Model export: what to watch for",
     ["For LoRA, log the MERGED fp32 model, not the adapter, so it loads anywhere with plain "
      "transformers and no PEFT dependency.",
      "Set an explicit ModelSignature, or downstream loaders have no contract.",
      "Known risk: an unresolved report of ONNX artifact download failing with OSError errno 95 "
      "to /Volumes, while non-ONNX flavors worked. ONNX is the obvious edge format, so test it "
      "for your own case rather than assuming.",
      "UC access control and lineage do NOT follow the exported artifact. Governance stops at "
      "the boundary.",
      "MLflow artifact reads are slow for large checkpoint sets."], None),

    (L_DIVIDER, "Appendix", None, None),

    (L_TITLE_ONLY, "Split 'no serverless' by clause", None,
     [["The requirement", "Verdict", "Why"],
      ["Compute must run in our own cloud account",
       "FOUNDED, not mitigable",
       "Serverless runs in the Databricks-managed plane; no customer-hosted variant exists"],
      ["Processing must stay in a named region, or off a specific cloud",
       "PARTLY founded",
       "The Geo enforcement setting is real but enforces Geo, not region; cross-region placement is not pinnable"],
      ["Our data must not leave our storage",
       "NOT founded",
       "Data, checkpoints and artifacts stay in your UC Volumes"]]),

    (L_CONTENT, "Notes on the serverless boundary",
     ["This applies to ALL serverless compute, not just GPU. The same plane runs CPU notebooks and jobs.",
      "Available mitigations: Geo enforcement, NCC / PrivateLink / egress controls, and keeping all "
      "data and checkpoints in UC Volumes",
      "Residual risks a reviewer can legitimately still raise: managed compute-plane ownership; "
      "unpinnable cross-region placement; Geo-level-only residency; transient GPU-host data with no "
      "published wipe SLA; the isolation whitepaper not being public",
      "FedRAMP High and DoD IL5 are documented as unsupported. Do not assert FedRAMP Moderate "
      "coverage without written confirmation."], None),

    (L_TITLE_ONLY, "Capability comparison", None,
     [["Dimension", "AI Runtime", "Classic DBR ML"],
      ["Accelerators", "A10, H100, B300", "T4, L4, A10G, L40S, V100, A100, H100"],
      ["Multi-GPU", "8xH100 / 8xB300 only", "Any multi-GPU node"],
      ["Multi-node", "CLI + torchrun, or Ray", "TorchDistributor, DeepSpeed, Ray-on-Spark"],
      ["Spark", "Spark Connect only", "Full cluster, Photon on CPU"],
      ["Data path", "UC Volumes, UCVolumeDataset", "Delta, Volumes, Mosaic Streaming"],
      ["Checkpoints", "PyTorch DCP to UC Volumes", "Volumes"],
      ["MLflow", "Auto system metrics + Logs tab", "Autologging, no auto system metrics"],
      ["Compliance", "Not FedRAMP High / IL5", "Standard profiles"]]),

    (L_CONTENT, "When classic is the right answer",
     ["Cloud account separation is required. This is the decisive case.",
      "Spark and GPU needed in the SAME cluster",
      "Ray-on-Spark, which is not available on serverless",
      "A library AI Runtime does not carry",
      "GPU types AI Runtime lacks: T4, L4, L40S, A100, V100",
      "A compliance profile AI Runtime does not cover",
      "Very long training runs where per-step throughput outweighs a 10-minute start. "
      "Our measured crossover was roughly 46 minutes of training-loop time (n=1, directional)."], None),

    (L_CONTENT, "The four costliest gotchas",
     ["1. environment.version: 6 silently gives an image with NO torch. Use 'databricks_ai_v6'.",
      "2. AI Runtime auto-retries deterministic failures. Set max_retries: 0 while developing.",
      "3. mlflow.pytorch.log_model plus AMP/Accelerate equals PicklingError. Unwrap and cast to fp32.",
      "4. Classic .py tasks have no implicit MLflow experiment. Set it in the script.",
      "Full list of 21 in FINDINGS.md, all found by running it rather than reading about it"], None),
]
