import random
from PIL import Image

import random
from PIL import Image

def shuffle_with_isolated_borders(image_path, output_path, margin=8):
    img = Image.open(image_path)
    width, height = img.size

    col_width = width // 4
    row_height = height // 4

    patches = []
    for i in range(4):
        for j in range(4):
            # Crop the original patch
            left = j * col_width
            top = i * row_height
            right = left + col_width
            bottom = top + row_height
            patch = img.crop((left, top, right, bottom))
            
            # CRITICAL STEP: Create an independent white frame around EVERY single patch
            # This isolates the 16x16 neighborhood window so it never sees a neighboring patch
            framed_w = col_width + (margin * 2)
            framed_h = row_height + (margin * 2)
            framed_patch = Image.new("RGB", (framed_w, framed_h), (255, 255, 255))
            framed_patch.paste(patch, (margin, margin))
            
            patches.append(framed_patch)

    # Image A: Save them in chronological order
    # (Do a run where you loop and paste them into a canvas without shuffling)
    
    # Image B: Random shuffle for the true test
    random.shuffle(patches)

    # Reassemble onto a large canvas
    canvas_w = (col_width + (margin * 2)) * 4
    canvas_h = (row_height + (margin * 2)) * 4
    final_canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    idx = 0
    for i in range(4):
        for j in range(4):
            x = j * (col_width + (margin * 2))
            y = i * (row_height + (margin * 2))
            final_canvas.paste(patches[idx], (x, y))
            idx += 1

    final_canvas.save(output_path)


# Example usage
shuffle_with_isolated_borders("mine1.jpg", "mine3.jpg")
