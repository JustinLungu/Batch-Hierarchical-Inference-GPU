# SML model registry with path
sml_model_registry = {
    "mobilenet_v3_large": {
        "path": "data/models/sml/mobilenet_v3_large_imagenet1k_v2.pth",
    },
    "efficientnet_b3": {
        "path": "data/models/sml/EfficientNet_B3_Weights_IMAGENET1K_V1.pth",
    },
    "resnet34": {
        "path": "data/models/sml/ResNet34_Weights_IMAGENET1K_V1.pth",
    }
}

# LML model registry with path
lml_model_registry = {
    "wide_resnet50_2": {
        "path": "data/models/lml/Wide_ResNet50_2_Weights_IMAGENET1K_V2.pth",
    },
    "efficientnet_v2_l": {
        "path": "data/models/lml/EfficientNet_V2_L_Weights_IMAGENET1K_V1.pth",
    },
    "vit_h_14": {
        "path": "data/models/lml/ViT_H_14_Weights_IMAGENET1K_SWAG_E2E_V1.pth",
    }
}