import torchvision.transforms.functional as TF


class CustomImageTransform:
    """Adjust an image in a custom way."""

    def __call__(self, img):
        img = TF.adjust_gamma(img, 1.6, 1.1)
        img = TF.adjust_brightness(img, 0.7)
        img = TF.adjust_saturation(img, 1.5)
        return img
