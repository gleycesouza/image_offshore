from PIL import Image
import numpy as np
import io
import cv2
from PIL import Image, ImageDraw, ImageFont
import re
import streamlit as st
import base64
import io
from PIL import Image
from typing import List, Tuple

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_page_bg(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def is_valid_depth(depth_str):
    """
    Verifica se o formato da profundidade é válido (ex: 0.1234).
    """
    return bool(re.match(r"^\d*\.\d+$", depth_str))

def format_depth(depth_str):
    try:
        return f"{float(depth_str):.2f}"
    except ValueError:
        return None

def get_tube_depths(num_tubes):
    tube_depths = []
    labels = ["A", "B", "C", "D"]
    tube_inputs = [st.text_input(f"Input tube {labels[i]} depth. Example: 0.1234", key=f"tube_depth_{i+1}") for i in range(num_tubes)]

    for tube_depth in tube_inputs:
        formatted_depth = format_depth(tube_depth)
        tube_depths.append(formatted_depth)

    return tube_depths

def process_image_top(bg_image, text_input):
    furo, testemunho, secao, amostra, TSA = process_text_input(text_input)
    image_circle = detect_circle_image(bg_image)
    resized_image = resize_image_circle(image_circle, 7.5, 600)
    border_added_image = add_border_circle(resized_image, 0.15, 0.479, 600)
    final_img = add_text_and_scale_circle(border_added_image, furo, testemunho, secao, amostra, 600)
    image_edited_name = furo + "_" + TSA + "_T"
    return final_img, image_edited_name

def process_image_bb(bg_image, text_input):
    furo, testemunho, secao, amostra, TSA = process_text_input(text_input)
    image_rectangle = detect_rectangle_image(bg_image)
    resized_image = resize_image_rectangle(image_rectangle, 1, 600)
    border_added_image = add_border_rectangle(resized_image, 0.15, 0.50, 0.50, 600)
    final_img = add_text_and_scale_rectangle(border_added_image, furo, testemunho, secao, amostra, 600)
    image_edited_name = furo + "_" + TSA + "_L"

    return final_img, image_edited_name

def process_image_group(num_tubes,tube_details,bg_image, text_input):
    furo, testemunho, _, _, _ = process_text_input(text_input)
    width, height = 3017, 6614
    left_margin, right_margin, top_margin, bottom_margin, margin = 300, 300, 472, 236, 215

    parts = cut_image_group(bg_image, 500, 15, num_tubes, dpi=600)
    template_img, tube_width, tube_height = create_template_with_scale_group(width, height, left_margin, right_margin, top_margin, bottom_margin, margin, parts)
    final_img = add_texts_group(template_img, width, height, left_margin, right_margin, top_margin, bottom_margin, tube_width, tube_height, margin, num_tubes, furo, testemunho, tube_details)
    image_edited_name = furo + "_" + testemunho + "_G"
    
    return final_img, image_edited_name

def detect_circle_image(image_bytes):
    nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

    # Convert image to grayscale
    gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect circles in the image using HoughCircles
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=50, maxRadius=1000)

    circles = np.uint16(np.around(circles))                                         # Change data type of the circles to int
    x, y, r = circles[0][0][1], circles[0][0][0], circles[0][0][2]-20               # Select the first detected circle
    cropped_image = image[x-r:x+r, y-r:y+r]                                         # Crop the image to fit the circle

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        x, y, r = circles[0]
        x, y, r = x * 2, y * 2, r * 2
        cropped_image = image[y-r:y+r, x-r:x+r]

        # Create a white circle mask
        mask = np.zeros(cropped_image.shape, dtype=np.uint8)
        mask = cv2.circle(mask, (r, r), r, (255, 255, 255), -1)
        result = cv2.bitwise_and(cropped_image, mask)

    # Convert the result to PIL Image object 
    result_image = Image.fromarray(result)
    return result_image

def detect_rectangle_image(image_bytes):
    nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                  # Convert image to grayscale
    blur = cv2.medianBlur(gray, 11)                                                 # Apply median blur to the image
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]    # Apply Otsu's threshold to the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Encontrar contornos que atendem às dimensões mínimas e selecionar o que tem o menor x
    valid_contours = [contour for contour in contours if 900 < cv2.boundingRect(contour)[2] < 5500 and 1300 < cv2.boundingRect(contour)[3] < 3000]

    if valid_contours:
        # Ordenar os contornos pelo menor valor de x
        leftmost_contour = sorted(valid_contours, key=lambda contour: cv2.boundingRect(contour)[0])[0]
        x, y, w, h = cv2.boundingRect(leftmost_contour)
        first_soil = image[y:y + h, x:x + w]

    # Convert the result to PIL Image object 
    result_image = Image.fromarray(first_soil)
    return result_image

def resize_image_circle(result_image, size_cm, dpi):
    # Calculate the size in pixels
    pixels = size_cm * dpi / 1.9227
    new_image = result_image.resize((int(pixels), int(pixels))) # Resize the image
    
    with io.BytesIO() as output:
        new_image.save(output, format="JPEG")
        img_bytes = output.getvalue()
    return img_bytes

def resize_image_rectangle(result_image, percent, dpi):
    # Obter o DPI original
    original_dpi = result_image.info.get("dpi", (dpi, dpi))

    # Calcular as novas dimensões em pixels
    new_width = int(result_image.width * percent)
    new_height = int(result_image.height * percent)

    # Redimensionar a imagem mantendo o DPI original
    resized_image = result_image.resize((new_width, new_height), resample=Image.LANCZOS)

    # Definir o DPI da imagem redimensionada
    resized_image.info["dpi"] = original_dpi

    with io.BytesIO() as output:
        resized_image.save(output, format="JPEG")
        img_bytes = output.getvalue()
    return img_bytes

def add_border_circle(result_image_resized, border_size_cm, border_size_top_cm, dpi):
    # Open the resized image as a PIL image
    #image = Image.open(image_path)
    image = Image.open(io.BytesIO(result_image_resized))

    # Set the desired border size in pixels
    right, left, bottom = (int(border_size_cm * dpi / 1.9227) for _ in range(3))
    top = int(border_size_top_cm * dpi / 1.9227)

    # Get the current image size
    width, height = image.size

    # Calculate the new size with the added borders
    new_width = width + right + left
    new_height = height + top + bottom

    # Create a new image with the new size and black background
    result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))

    # Paste the original image on the new image with the added borders
    result.paste(image, (left, top))

    return result

def add_border_rectangle(result_image_resized, border_size_side_cm, border_top_cm, border_bot_cm, dpi):
    # Open the resized image as a PIL image
    #image = Image.open(image_path)
    image = Image.open(io.BytesIO(result_image_resized))

    # Set the desired border size in pixels
    right = int(border_size_side_cm * dpi / 2.54)
    left = int(border_size_side_cm * dpi / 2.54)
    top = int(border_top_cm * dpi / 2.54)
    bottom = int(border_bot_cm * dpi / 2.54)

    # Get the current image size
    width, height = image.size

    # Calculate the new size with the added borders
    new_width = width + right + left
    new_height = height + top + bottom

    # Create a new image with the new size and black background
    result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))

    # Paste the original image on the new image with the added borders
    result.paste(image, (left, top))

    return result

def add_text_and_scale_circle(result, furo, testemunho, secao, amostra, dpi):
    # Open an Image
    img = result

    # Create an ImageDraw object
    draw = ImageDraw.Draw(img)

    # Define custom font style and font size
    myFont_a = ImageFont.truetype('./font/ARIAL.TTF', 86)
    myFont = ImageFont.truetype('./font/ARIALBD.TTF', 86)

    # Define text to be added to the image
    texts = [
        (furo, 20, 20),
        (testemunho, 1790, 20),
        (secao, 2030, 20),
        (amostra, 2230, 20)
    ]
    # Add Text to an image
    for value, x, y in texts:
        draw.text((x, y), value, font=myFont, fill='white')

    # Draw a rectangle with the specified coordinates
    x1, y1, x2, y2 = 1939, 2400, 2250, 2500
    rec = ImageDraw.Draw(img)
    rec.rectangle((x1, y1, x2, y2), fill='white')

    # Scale
    draw.text((2000, 2406), "1 cm", font=myFont_a, fill='black')
 
    # Save the edited image
    buf = io.BytesIO()
    img.save(buf, format='jpeg', dpi=(dpi, dpi), quality=95)
    
    return buf.getvalue()

def add_text_and_scale_rectangle(result, furo, testemunho, secao, amostra, dpi):
    # Open an Image
    img = result

    # Create an ImageDraw object
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Define custom font style and font size
    myFont_a = ImageFont.truetype('./font/ARIAL.TTF', 50)
    myFont = ImageFont.truetype('./font/ARIALBD.TTF', 65)

    # Define text to be added to the image
    texts = [
        (furo, 30, 30),
        (testemunho, width-460, 30),
        (secao, width-300, 30),
        (amostra, width-170, 30)
    ]
    # Add Text to an image
    for value, x, y in texts:
        draw.text((x, y), value, font=myFont, fill='white')

    # Definindo a altura e a largura do retângulo
    rect_width = 127  # 2 * 118
    rect_height = 127  # 2 * 110

    # Calculando as coordenadas com base na altura e largura do retângulo
    x1 = (width / 2) - (rect_width / 2)
    y1 = (height-40) - (rect_height / 2)
    x2 = (width / 2) + (rect_width / 2)
    y2 = (height-80) + (rect_height / 2)

    # Desenhar o retângulo com as coordenadas especificadas
    rec = ImageDraw.Draw(img)
    rec.rectangle((x1, y1, x2, y2), fill='white')

    # Scale
    draw.text(((width/2)-60, height-90), "1 cm", font=myFont_a, fill='black')
 
    # Save the edited image
    buf = io.BytesIO()
    img.save(buf, format='jpeg', dpi=(dpi, dpi), quality=95)
    
    return buf.getvalue()

def process_text_input(text_input):

    text_input_list = text_input.split(",")
    furo = text_input_list[0]
    TSA = text_input_list[1]

    testemunho = re.split(re.compile(r'[ABC]'), text_input_list[1])[0]
    # Verifica se há uma seção ou amostra presente, caso contrário, define como 0
    secao = re.findall(re.compile(r'[ABC]'), text_input_list[1])[0]
    amostra = re.split(re.compile(r'[ABC]'), text_input_list[1])[1]

    secao = 'S-' + secao
    
    if len(testemunho) == 1:
        testemunho = 'T-0' + testemunho
    else:
        testemunho = 'T-' + testemunho

    if len(amostra) == 1:
        amostra = 'A-0' + amostra
    else:
        amostra = 'A-' + amostra

    return (furo, testemunho, secao, amostra, TSA)

def cut_image_group(image_bytes: str, part_height: int, margin: int, total_parts: int, dpi: int) -> List[Image.Image]:
    """
    Corta a imagem em partes e retorna uma lista de objetos PIL Image representando cada parte.
    """
    nparr = np.frombuffer(image_bytes.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

    # Obter as dimensões da imagem
    h, w, _ = image.shape

    # Calcular a coordenada y para cortar a imagem
    half_height = h // 2
    extra_height = 100
    new_top_height = half_height - extra_height
    bottom_half = image[new_top_height:, :]

    # Garantir que o número de partes e margens caibam na altura da imagem
    if total_parts * part_height - (total_parts - 1) * margin > bottom_half.shape[0]:
        raise ValueError("The total height of the margined parts exceeds the image height.")

    # Inicializar a altura de corte
    start_y = 0
    parts = []

    # Cortar a imagem em partes com margem
    for i in range(total_parts):
        end_y = start_y + part_height
        if i < total_parts - 1:
            # Para as partes intermediárias, adicionar uma margem
            end_y += margin
        part = bottom_half[start_y:end_y, :]

        # Converter a parte cortada de OpenCV para PIL
        part_pil = Image.fromarray(cv2.cvtColor(part, cv2.COLOR_BGR2RGB))

        # Redefinir a resolução da imagem
        part_pil.info['dpi'] = (dpi, dpi)

        # Adicionar a parte à lista
        parts.append(part_pil)

        # Atualizar a altura de início para a próxima parte
        start_y = end_y - margin

    return parts

def create_template_with_scale_group(width: int, height: int, left_margin: int, right_margin: int, top_margin: int, bottom_margin: int, margin: int, tube_images: List[Image.Image]) -> Tuple[Image.Image, int, int]:
    tube_width = (width - (left_margin + right_margin + 3 * margin)) // 4  # Ajustado para 4 tubos
    tube_height = height - (top_margin + bottom_margin)

    # Criar a imagem base
    img = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Desenhar os retângulos para os tubos e colar as imagens
    for i in range(4):  # Loop para desenhar 4 tubos
        x1 = left_margin + i * (tube_width + margin)
        x2 = x1 + tube_width
        y1 = top_margin
        y2 = top_margin + tube_height
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2, fill=(255, 255, 255))

        # Adicionar a imagem do tubo se disponível
        if i < len(tube_images):
            tube_img = tube_images[i]
            tube_img = tube_img.rotate(270, expand=1)  # Rotacionar a imagem em 90°
            tube_img = tube_img.resize((tube_width, tube_height))  # Redimensionar para caber no retângulo
            img.paste(tube_img, (x1, y1))

    # Adicionar a escala intercalada
    scale_width = 160
    for i in range(10):
        y = top_margin + i * (tube_height // 10)
        color = (169, 169, 169) if i % 2 == 0 else (255, 255, 255)  # cinza
        draw.rectangle([(width - 70 - scale_width, y),
                        (width - 70, y + (tube_height // 10))], fill=color)
        if i == 0:
            # Criar uma nova imagem para o texto rotacionado
            font = ImageFont.truetype('./font/ARIALBD.TTF', 150)
            txt_img = Image.new('RGBA', (font.getsize(f"10 cm")[0], font.getsize(f"10 cm")[1]), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((0, 0), f"10 cm", fill=(255, 0, 0), font=font)
            txt_img = txt_img.rotate(90, expand=1)
            # Posição ajustada para colar o texto rotacionado dentro do retângulo
            text_x = width - 74 - scale_width + (scale_width - txt_img.width) // 2
            text_y = y + (tube_height // 10 - txt_img.height) // 2
            img.paste(txt_img, (text_x, text_y), txt_img)

    return img, tube_width, tube_height

def add_texts_group(template_img: Image.Image, width: int, height: int, left_margin: int, right_margin: int, top_margin: int, bottom_margin: int, tube_width: int, tube_height: int, margin: int, num_tubos: int, borehole: str, testemunho: str, depths: List[str]) -> Image.Image:
    draw = ImageDraw.Draw(template_img)
    white = (255, 255, 255)
    font = ImageFont.truetype('./font/ARIALBD.TTF', 150)
    draw.text((80*4, 50), borehole, fill=white, font=font)
    draw.text((width - margin * 3, 50), testemunho, fill=white, font=font)
    draw.text((80, 500), "T", fill=white, font=font)
    draw.text((80, 500 + 200), "O", fill=white, font=font)
    draw.text((80, 500 + 200 * 2), "P", fill=white, font=font)
    draw.text((80, 500 + 200 * 3), "O", fill=white, font=font)

    # Textos para cada tubo
    for i in range(num_tubos):
        draw.text((left_margin + i * margin + ((tube_width) * (i + 0.15)), 70 * 4), depths[i], fill=white, font=font)

    # Textos no final de cada tubo
    labels = ["S-A", "S-B", "S-C", "S-D"]  # Supondo que você tenha até 4 tubos, pode ajustar conforme necessário
    for i in range(num_tubos):
        draw.text((left_margin + i * margin + ((tube_width) * (i + 0.25)), height - 200), labels[i], fill=white, font=font)

        # Save the edited image
    buf = io.BytesIO()
    template_img.save(buf, format='jpeg', dpi=(600, 600), quality=95)
    
    return buf.getvalue()

    # return template_img