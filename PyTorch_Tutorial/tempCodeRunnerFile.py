random_image_size_tensor = torch.rand(size=(224, 224, 3)) # 3 channels (RGB) of 224x224 pixels
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)  