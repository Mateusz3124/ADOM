from PIL import Image, ImageEnhance, ImageFilter

# Wczytaj oryginalne zdjęcie twarzy
img = Image.open("mine1.jpg").convert("RGB")

# 1. Ekstremalne wyostrzenie (znajduje mikro-krawędzie, których SIFT nienawidzi)
img_sharp = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
img_sharp = img_sharp.filter(ImageFilter.SHARPEN)

# 2. Drastyczne podbicie kontrastu (zamienia miękkie cienie w twarde, sztuczne krawędzie)
enhancer = ImageEnhance.Contrast(img_sharp)
img_broken_gradients = enhancer.enhance(4.5)  # Bardzo mocny kontrast

# 3. Zapisz obraz
img_broken_gradients.save("mine6.jpg")
print("Wygenerowano obraz z zaburzonymi gradientami!")
