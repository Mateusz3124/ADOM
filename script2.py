import numpy as np
from PIL import Image, ImageDraw

# Wczytanie oryginalnego zdjęcia twarzy
img = Image.open("mine1.jpg").convert("RGB")
width, height = img.size

# --- TEST 1: AGRESYWNY SZUM (Wymuszenie błędu kwantyzacji) ---
img_array = np.array(img, dtype=np.float32)
# Podnosimy siłę szumu z 15 do 45 (twarz wciąż widać, ale deskryptory zwariują)
mocny_szum = np.random.normal(0, 45, img_array.shape) 
noisy_array = np.clip(img_array + mocny_szum, 0, 255).astype(np.uint8)
img_szum = Image.fromarray(noisy_array)
img_szum.save("mine4.jpg")


# --- TEST 2: TEKSTURA NA CAŁYM OBRAZIE (Paski zasłaniające twarz) ---
img_paski_caly = img.copy()
draw = ImageDraw.Draw(img_paski_caly)

# Rysujemy gęste paski na całej szerokości i wysokości obrazu
# To symuluje twarz za żaluzjami lub mocny wzór makijażu/kamuflażu
for y in range(0, height, 6):
    draw.line([(0, y), (width, y)], fill=(0, 0, 0), width=2)
    draw.line([(0, y+3), (width, y+3)], fill=(255, 255, 255), width=2)

img_paski_caly.save("mine5.jpg")

print("Nowe, trudniejsze obrazy testowe zostały wygenerowane!")
