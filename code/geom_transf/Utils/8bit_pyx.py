from PIL import Image

# Load image
im = Image.open('C:\\Users\\Eros Bignardi\\projectCV\\data\\train\\cloth\\12129734_1.jpg')

# Convert to palette mode and save. Method 3 is "libimagequant"
im.quantize(colors=256, method=3).save('result.jpg')


im.save("C:\\Users\\Eros Bignardi\\projectCV\\result.jpg")
