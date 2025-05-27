
# SAEs Are Good for Steering - If You Select the Right Features

[Website](https://technion-cs-nlp.github.io/saes-are-good-for-steering/) | [Paper](https://arxiv.org/abs/2505.20063)

Sparse Autoencoders (SAEs) have been proposed as an unsupervised approach to learn a decomposition of a model's latent space. This enables useful applications such as steering - influencing the output of a model towards a desired concept - without requiring labeled data. Current methods identify SAE features to steer by analyzing the input tokens that activate them. However, recent work has highlighted that activations alone do not fully describe the effect of a feature on the model's output. In this work, we draw a distinction between two types of features: input features, which mainly capture patterns in the model's input, and output features, which have a human-understandable effect on the model's output. We propose input and output scores to characterize and locate these types of features, and show that high values for both scores rarely co-occur in the same features. These findings have practical implications: after filtering out features with low output scores, we obtain 2-3x improvements when steering with SAEs, making them competitive with supervised methods.


## Requirements
```
  pip install -r requirements
  pip install accelerate
  pip install sae-lens
```

## Code
Output Scores

```
python ./src/output_score.py --model_type=<model_type> --features_file=<features_json> --cache_path=<filename_to_load_and_save>
```

Input Scores

First, download feature data from [Neuronpedia](https://www.neuronpedia.org/api-doc#tag/features/POST/api/activation/new). 
Then run:

```
python ./src/input_score.py --model_type=<model_type> --features_file=<features_json> --cache_path=<filename_to_load_and_save> --feature_data_path=<path>
```

## Data
We provide the exact features used in our experiments, the generated texts for each steering factor and feature, the best steering factor for each feature.
For Concept500 features we additionally provide the 10 sampled instructions per feature and the concept, instruct and fluency scores computed by an external LLM.

 

