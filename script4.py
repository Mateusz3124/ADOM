from PIL import Image, ImageEnhance, ImageFilter

# Wczytaj oryginalne zdjęcie twarzy
img = Image.open("mine1.jpg").convert("RGB")

# 1. Silne rozmycie (usuwa mikro-krawędzie i detale, przez co SIFT nie ma na czym się zahaczyć)
img_blurred = img.filter(ImageFilter.GaussianBlur(radius=5))

# 2. Drastyczne obniżenie kontrastu (zamienia wyraźne przejścia w jednolitą, szarą masę)
# enhancer = ImageEnhance.Contrast(img_blurred)
# img_flat_gradients = enhancer.enhance(0.2)  # Bardzo niski kontrast (wartość blisko zera)

# 3. Zapisz obraz
img_blurred.save("mine8.jpg")
print("Wygenerowano obraz z całkowicie spłaszczonymi i rozmytymi gradientami!")