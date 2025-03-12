import os
import torch
import cv2
import numpy as np
from manga_ocr import MangaOcr
from PIL import Image
from craft_text_detector import Craft
from ultralytics import YOLO
import openai
import json
from sklearn.cluster import KMeans


class TextBlock:
    def __init__(self, xyxy: list[float], text: str = "", translation: str = ""):
        self.xyxy = [float(x) for x in xyxy]  # Ensure all values are float
        if len(self.xyxy) != 4:
            raise ValueError("xyxy must contain exactly 4 values")
        self.text = text
        self.translation = translation

class TextBlockDetector:
    def __init__(self, bubble_model_path: str, text_seg_model_path: str, text_detect_model_path: str, device: str):
        self.bubble_detection = YOLO(bubble_model_path)
        self.text_segmentation = YOLO(text_seg_model_path)
        self.text_detection = YOLO(text_detect_model_path)
        self.device = device

    def detect(self, img: np.ndarray) -> list[TextBlock]:
        h, w, _ = img.shape
        size = (h, w) if h >= w * 5 else 1024
        det_size = (h, w) if h >= w * 5 else 640

        bble_detec_result = self.bubble_detection(img, device=self.device, imgsz=size, conf=0.1)[0]
        txt_seg_result = self.text_segmentation(img, device=self.device, imgsz=size, conf=0.1, verbose=False)[0]
        txt_detect_result = self.text_detection(img, device=self.device, imgsz=det_size, conf=0.2, verbose=False)[0]

        combined = self.combine_results(img, bble_detec_result, txt_seg_result, txt_detect_result)

        blk_list = [TextBlock(txt_bbox, bble_bbox, txt_class)
                    for txt_bbox, bble_bbox, txt_class in combined]
        
        return blk_list
    
    def combine_results(self, img, bubble_detec_results, text_seg_results, text_detect_results):
        bubble_bounding_boxes = bubble_detec_results.boxes.xyxy.cpu().numpy()

        # save the bubble detection result
        for box in bubble_bounding_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imwrite('bubble_detection.jpg', img)
        
        seg_text_bounding_boxes = text_seg_results.boxes.xyxy.cpu().numpy()
        detect_text_bounding_boxes = text_detect_results.boxes.xyxy.cpu().numpy()

        # print(seg_text_bounding_boxes)
        # Show the segmentation on the image
        for box in seg_text_bounding_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imwrite('segmentation.jpg', img)

        # Show the detection on the image
        for box in detect_text_bounding_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.imwrite('detection.jpg', img)

        text_bounding_boxes = self.merge_bounding_boxes(seg_text_bounding_boxes, detect_text_bounding_boxes)

        # Show the merged result on the image
        for box in text_bounding_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.imwrite('merged.jpg', img)

        raw_results = []
        text_matched = [False] * len(text_bounding_boxes)

        if len(text_bounding_boxes) > 0:
            for txt_idx, txt_box in enumerate(text_bounding_boxes):
                for bble_box in bubble_bounding_boxes:
                    if self.does_rectangle_fit(bble_box, txt_box):
                        raw_results.append((txt_box, bble_box, 'text_bubble'))
                        text_matched[txt_idx] = True
                        break
                    elif self.do_rectangles_overlap(bble_box, txt_box):
                        raw_results.append((txt_box, bble_box, 'text_free'))
                        text_matched[txt_idx] = True
                        break

                if not text_matched[txt_idx]:
                    raw_results.append((txt_box, None, 'text_free'))

        return raw_results

    @staticmethod
    def merge_bounding_boxes(seg_boxes, detect_boxes):
        merged_boxes = []
        for seg_box in seg_boxes:
            running_box = seg_box
            for detect_box in detect_boxes:
                if TextBlockDetector.does_rectangle_fit(running_box, detect_box):
                    continue
                if TextBlockDetector.do_rectangles_overlap(running_box, detect_box, 0.02):
                    running_box = TextBlockDetector.merge_boxes(running_box, detect_box)
            merged_boxes.append(running_box)

        for detect_box in detect_boxes:
            if not any(TextBlockDetector.do_rectangles_overlap(detect_box, merged_box, 0.1) or 
                       TextBlockDetector.does_rectangle_fit(merged_box, detect_box) for merged_box in merged_boxes):
                merged_boxes.append(detect_box)

        return np.array(merged_boxes)

    @staticmethod
    def do_rectangles_overlap(rect1, rect2, iou_threshold: float = 0.2) -> bool:
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[2], rect2[2])
        y2 = min(rect1[3], rect2[3])
        
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
        union_area = rect1_area + rect2_area - intersection_area
        
        iou = intersection_area / union_area if union_area != 0 else 0
        return iou >= iou_threshold

    @staticmethod
    def does_rectangle_fit(bigger_rect, smaller_rect):
        return (smaller_rect[0] >= bigger_rect[0] and smaller_rect[1] >= bigger_rect[1] and
                smaller_rect[2] <= bigger_rect[2] and smaller_rect[3] <= bigger_rect[3])

    @staticmethod
    def merge_boxes(box1, box2):
        return [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3])
        ]

def load_craft_model():
    craft = Craft(output_dir=None, crop_type="poly", cuda=torch.cuda.is_available())
    return craft

def load_manga_ocr_model():
    return MangaOcr()

def load_text_block_detector():
    return TextBlockDetector(
        bubble_model_path='models/comic-speech-bubble-detector.pt',
        text_seg_model_path='models/comic-text-segmenter.pt',
        text_detect_model_path='models/manga-text-detector.pt',
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

def find_character_color(image, char_box):
    x, y, w, h = char_box
    char_image = image[y:y+h, x:x+w]
    
    # Convert to grayscale
    gray = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Dilate the edges to connect nearby edges
    kernel = np.ones((3,3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Create a mask from the edges
    mask = dilated_edges > 0
    
    # Get colors of the pixels on the edges
    edge_colors = char_image[mask]
    
    if len(edge_colors) == 0:
        return None
    
    # Reshape the colors for K-means clustering
    edge_colors = edge_colors.reshape(-1, 3)
    
    # Perform K-means clustering on the colors
    kmeans = KMeans(n_clusters=2, n_init=10)
    kmeans.fit(edge_colors)
    
    # Get the two most dominant colors
    colors = kmeans.cluster_centers_.astype(int)
    
    # Calculate the luminance of each color
    luminances = np.dot(colors, [0.299, 0.587, 0.114])
    
    # The color with lower luminance is likely to be the text color
    text_color = colors[np.argmin(luminances)]
    
    return tuple(text_color)

def rgb_to_hex(b, g, r):
    # convert bgr to rgb
    rgb = (r, g, b)
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def detect_text_manga_ocr(image, manga_ocr_model, text_block_detector, craft_model):
    # Detect text blocks
    text_blocks = text_block_detector.detect(image)
    
    # Perform OCR on each text block
    detected_boxes = []
    detected_texts = []
    detected_colors = []
    for block in text_blocks:
        x1, y1, x2, y2 = map(int, block.xyxy)
        text_image = image[y1:y2, x1:x2]
        text = manga_ocr_model(Image.fromarray(text_image))
        
        if text.strip():  # If text is not empty
            # Use CRAFT to detect a single character within this text block
            prediction_result = craft_model.detect_text(text_image)
            if len(prediction_result['boxes']) > 0:
                char_box = prediction_result['boxes'][0]  # Take the first detected character
                char_x1, char_y1 = map(int, char_box[0])
                char_x2, char_y2 = map(int, char_box[2])
                char_w, char_h = char_x2 - char_x1, char_y2 - char_y1
                
                color = find_character_color(text_image, (char_x1, char_y1, char_w, char_h))
                hex_color = rgb_to_hex(*color) if color else None

                detected_boxes.append([x1, y1, x2, y2])
                detected_texts.append(text)
                detected_colors.append(color)
                print(f"Detected text: {text}, Color: {hex_color}")
            else:
                print(f"No character detected by CRAFT for text: {text}")
        else:
            print(f"No text detected in box at {x1, y1, x2, y2}")

    return detected_boxes, detected_texts, detected_colors

def detect_text_craft(image, craft_model):
    prediction_result = craft_model.detect_text(image)
    return prediction_result['boxes']


def translate_with_chatgpt(texts, target_languages):
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    prompt = f"Translate the following texts to {', '.join(target_languages)}. Maintain the context and style of manga dialogue. Return the results as a JSON array where each object has 'original' and 'translations' keys. The 'translations' key should be an object with language codes as keys and translations as values.\n\n"
    for i, text in enumerate(texts):
        prompt += f"{i+1}. {text}\n"

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a professional manga translator."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response['choices'][0]['message']['content']
    
    # Remove markdown formatting if present
    content = content.strip('`').strip()
    if content.startswith('json'):
        content = content[4:].strip()

    try:
        # Find the start and end of the JSON array
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end != -1:
            json_content = content[start:end]
            translations = json.loads(json_content)
        else:
            raise ValueError("Could not find valid JSON array in the response")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error decoding JSON: {e}")
        print(f"Raw content: {content}")
        translations = [{"original": text, "translations": {lang: "Translation error" for lang in target_languages}} for text in texts]

    return translations

def detect_text(image, ocr_model, ocr_type, text_block_detector=None):
    if ocr_type == 'craft':
        return detect_text_craft(image, ocr_model)
    elif ocr_type == 'manga_ocr':
        return detect_text_manga_ocr(image, ocr_model, text_block_detector)
    else:
        raise ValueError("Invalid OCR type. Choose 'craft' or 'manga_ocr'.")

def remove_text(image, boxes):
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    for box in boxes:
        cv2.fillPoly(mask, [np.array(box, dtype=np.int32)], (0, 0, 0))
    
    # Inpainting
    inpainted_image = cv2.inpaint(image, 255 - mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return inpainted_image

def visualize_detections(image, boxes, colors):
    vis_image = image.copy()
    for box, color in zip(boxes, colors):
        # Ensure we have 4 points to draw a rectangle
        if len(box) == 4 and type(box[0]) != np.ndarray:
            x1, y1, x2, y2 = map(int, box)
            if color is not None:
                try:
                    bgr_color = (int(color[0]), int(color[1]), int(color[2]))
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), bgr_color, 2)
                except (IndexError, ValueError, TypeError):
                    print(f"Invalid color value: {color}. Using default color.")
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Default to green if color is invalid
            else:
                print("Color is None. Using default color.")
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Default to green if color is None
        elif len(box) == 4 and type(box[0]) == np.ndarray:
            pts = np.array(box).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_image, [pts], True, color, 2)
        elif len(box) == 2:
            # If we only have 2 points, assume it's top-left and bottom-right
            (x1, y1), (x2, y2) = box
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return vis_image

class MangaTranslator:
    def __init__(self, bubble_model_path: str):
        # Initialize text detection and translation components
        self.text_detector = TextBlockDetector(
            bubble_model_path=bubble_model_path,
            text_seg_model_path='models/comic-text-segmenter.pt',
            text_detect_model_path='models/manga-text-detector.pt',
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.manga_ocr = MangaOcr()
        self.craft_model = Craft(output_dir=None, crop_type="poly", cuda=torch.cuda.is_available())

    def process_image(self, image_path: str, target_languages: list[str]) -> tuple[np.ndarray, list]:
        """
        Process image and return both the image and translation data
        Saves translation to a JSON file based on the image filename
        """
        # Generate output JSON path based on input image path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_json_path = f'translation_result_{base_name}.json'
        
        return self.process_image_with_custom_output(image_path, target_languages, output_json_path)
    
    def process_image_with_custom_output(self, image_path: str, target_languages: list[str], 
                                         output_json_path: str) -> tuple[np.ndarray, list]:
        """
        Process image with a custom output JSON path
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        if len(image.shape) == 3 and image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        boxes, texts, colors = detect_text_manga_ocr(
            image, 
            self.manga_ocr, 
            self.text_detector, 
            self.craft_model
        )
        
        if not texts:
            print(f"No text detected in {image_path}")
            translation_data = []
        else:
            translations = translate_with_chatgpt(texts, target_languages)
            
            translation_data = []
            for box, original, translation, color in zip(boxes, texts, translations, colors):
                translation_data.append({
                    "bounding_box": box,
                    "original_text": original,
                    "content": translation['translations'],
                    "text_color": rgb_to_hex(*color) if color else None
                })

        # Save the translation data to the specified JSON file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(translation_data, f, ensure_ascii=False, indent=2)
            
        print(f"Saved translation data to {output_json_path}")
        
        return image, translation_data

