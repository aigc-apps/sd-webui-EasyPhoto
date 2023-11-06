
# Uses
_Note: This section is originally taken from the [Stable Diffusion v2 model card](https://huggingface.co/stabilityai/stable-diffusion-2), but applies in the same way to Zero-1-to-3._

## Direct Use 
The model is intended for research purposes only. Possible research areas and tasks include:

- Safe deployment of large-scale models.
- Probing and understanding the limitations and biases of generative models.
- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.
- Research on generative models.

Excluded uses are described below.

### Misuse, Malicious Use, and Out-of-Scope Use
The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

#### Out-of-Scope Use
The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

#### Misuse and Malicious Use
Using the model to generate content that is cruel to individuals is a misuse of this model. This includes, but is not limited to:

- Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.
- Intentionally promoting or propagating discriminatory content or harmful stereotypes.
- Impersonating individuals without their consent.
- Sexual content without consent of the people who might see it.
- Mis- and disinformation
- Representations of egregious violence and gore
- Sharing of copyrighted or licensed material in violation of its terms of use.
- Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.

## Limitations and Bias

### Limitations

- The model does not achieve perfect photorealism.
- The model cannot render legible text.
- Faces and people in general may not be parsed or generated properly.
- The autoencoding part of the model is lossy.
- Stable Diffusion was trained on a subset of the large-scale dataset [LAION-5B](https://laion.ai/blog/laion-5b/), which contains adult, violent and sexual content. To partially mitigate this, Stability AI has filtered the dataset using LAION's NSFW detector.
- Zero-1-to-3 was subsequently finetuned on a subset of the large-scale dataset [Objaverse](https://objaverse.allenai.org/), which might also potentially contain inappropriate content. To partially mitigate this, our demo applies a safety check to every uploaded image.

### Bias
While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases. 
Stable Diffusion was primarily trained on subsets of [LAION-2B(en)](https://laion.ai/blog/laion-5b/), which consists of images that are limited to English descriptions. 
Images and concepts from communities and cultures that use other languages are likely to be insufficiently accounted for. 
This affects the overall output of the model, as Western cultures are often overrepresented. 
Stable Diffusion mirrors and exacerbates biases to such a degree that viewer discretion must be advised irrespective of the input or its intent.


### Safety Module
The intended use of this model is with the [Safety Checker](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) in Diffusers. 
This checker works by checking model inputs against known hard-coded NSFW concepts.
Specifically, the checker compares the class probability of harmful concepts in the embedding space of the uploaded input images. 
The concepts are passed into the model with the image and compared to a hand-engineered weight for each NSFW concept.

## Citation
```
@misc{liu2023zero1to3,
      title={Zero-1-to-3: Zero-shot One Image to 3D Object}, 
      author={Ruoshi Liu and Rundi Wu and Basile Van Hoorick and Pavel Tokmakov and Sergey Zakharov and Carl Vondrick},
      year={2023},
      eprint={2303.11328},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
