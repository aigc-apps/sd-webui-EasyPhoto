## Tips if not getting expected results :detective:
1. If you are trying out images of humans, especially faces, note that it's unfortunately not the intended use cases. We would encourage to try out images of everyday objects or even artworks.
2. If some part of the object is missing, check the interactive angle visualization pane (top right) where you can find a panel of the actual input image to the model after preprocessing steps and see if the segmented image contains the entire object you are trying to visualize.
3. The model is probabilistic, therefore, if the number of samples is selected to be bigger than 1 and results look different, that's expected as the model tries to predict a diverse set of possibilities given the input image and the specified camera viewpoint.
4. Under "advanced options", you can tune two parameters as you can typically find in other stable diffusion demos as well:
	 - Diffusion Guidance Scale defines how much you want the model to respect the input information (image + angles). Higher scale typically leads to less diversity and higher image distortion.
	 - Number of diffusion inference steps controls the number of diffusion steps is applied to generate each image. Usually the higher the better with a diminishing return.

Have fun!

A model card can be found here: https://github.com/cvlab-columbia/zero123/blob/main/zero123/uses.md