# Inference Usage Instruction

Merlin can be run by instantiating the model in PyTorch. Merlin weights are also publicly available on [HuggingFace](https://huggingface.co/stanfordmimi/Merlin).
- Image/Text contrastive embeddings
- Image-only embeddings (provide similar functionality to Google CT Foundation)

For a better understanding of the phenotypes and their associated PheWAS attributes, please refer to the [phenotypes](phenotypes.csv) file.

**Please see the [demo](demo.py) for programmatic examples.**

#### Image/Text contrastive embeddings

To get the image/text constrastive embeddings for inference, the breakdown is as follows:

```python
import torch
from merlin import Merlin

model = Merlin()
model.eval()
model.cuda()

for batch in dataloader:
    outputs = model(
        batch["image"].to(device), 
        batch["text"]
        )
```

where `outputs` is a tuple:
- `outputs[0]` : returns the constrative image embeddings (shape: [1, 512])
- `outputs[1]` : returns the phenotype prediction (shape: [1, 1692])
- `outputs[2]` : returns the constrative text embeddings (shape: [1, 512])

#### Image-only embeddings

```python
import torch
from merlin import Merlin

model = Merlin(ImageEmbedding=True)
model.eval()
model.cuda()

for batch in dataloader:
    outputs = model(
        batch["image"].to(device), 
        )
```

where `outputs` is a tuple:
- `outputs[0]` : returns the image embeddings (shape: [1, 2048])


## 👨‍💻 Merlin Finetuning

Since both Merlin’s model architecture and pretrained weights are provided, Merlin allows for straightforward finetuning in PyTorch VLM and vision-only pipelines. Additionally, Merlin was trained on a single NVIDIA A6000 GPU (with a Vision-Language batch size of 18), meaning finetuning can be performed even in compute-constrained environments.

Merlin supports both Image/Text and Image-only finetuning. To perform finetuning, simply remove the following lines of code and train on your data:
~~`model.eval()`~~  
~~`model.cuda()`~~  

For compute-efficient finetuning, we recommend using mixed-precision training and gradient accumulation.