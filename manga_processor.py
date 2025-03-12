import torch
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import os
import json
from huggingface_hub.constants import HF_HUB_CACHE
from translator import MangaTranslator
from controlnet_union import ControlNetModel_Union
from huggingface_hub import hf_hub_download
from diffusers.models.model_loading_utils import load_state_dict
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers import AutoencoderKL
from krita_lib import KritaDocument, TextStyle, ShapeStyle, ShapeLayer, Rectangle, Circle, Path, ShapeGroup, PaintLayer, LayerStyle
# from realesrgan import RealESRGANer
# from basicsr.archs.rrdbnet_arch import RRDBNet



class MangaPageProcessor:
    def __init__(self, bubble_model_path, sfx_model_path=None, overlap_threshold=0.05):
        self.cache_dir = HF_HUB_CACHE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.overlap_threshold = overlap_threshold
        self.min_font_size = 24
        self.max_font_size = 45
        self.min_text_free_font_size = 40
        self.max_text_free_font_size = 55
        self.max_scale_factor = 1.5  # Maximum bubble scale factor
        self.contour_area_threshold = 100

        self._load_models(bubble_model_path, sfx_model_path)
        self.setup_inpainting()

    def _load_models(self, bubble_model_path, sfx_model_path):
        """Load detection models once"""
        from ultralytics import YOLO  # Importing here prevents circular imports
        from segment_anything import SamPredictor, sam_model_registry

        print("🔍 Loading bubble detection model...")
        self.bubble_detector = YOLO(bubble_model_path)

        # Load SFX detector if path is provided
        self.sfx_detector = None
        if sfx_model_path and os.path.exists(sfx_model_path):
            print(f"🎶 Loading SFX detector from {sfx_model_path}")
            self.sfx_detector = YOLO(sfx_model_path)

        # Load Segment Anything model
        print("📌 Loading SAM model...")
        sam_checkpoint_path = hf_hub_download(
            "HCMUE-Research/SAM-vit-h", "sam_vit_h_4b8939.pth"
        )
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
        sam.to(self.device)
        self.sam_predictor = SamPredictor(sam)

    def setup_inpainting(self):
        """Setup inpainting pipeline for test-free areas"""
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Load ControlNet
        config_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="config_promax.json",
        )
        config = ControlNetModel_Union.load_config(config_file)
        controlnet_model = ControlNetModel_Union.from_config(config)
        model_file = hf_hub_download(
            "xinsir/controlnet-union-sdxl-1.0",
            filename="diffusion_pytorch_model_promax.safetensors",
        )
        state_dict = load_state_dict(model_file)
        model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
            controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
        )
        model.to(device=self.device, dtype=dtype)

        # model.enable_model_cpu_offload()

        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=dtype
        ).to(self.device)

        # Initialize pipeline
        self.inpaint_pipe = StableDiffusionXLFillPipeline.from_pretrained(
            "John6666/holy-mix-illustriousxl-vibrant-anime-checkpoint-v1-sdxl",
            torch_dtype=dtype,
            vae=vae,
            controlnet=model,
        ).to(self.device)

        self.inpaint_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.inpaint_pipe.scheduler.config
        )

        # Pre-encode prompt for inpainting
        prompt = "high quality"
        self.prompt_embeds = self.inpaint_pipe.encode_prompt(prompt, self.device, True)

    def visualize_detections(self, image, bubbles, merged_bubbles, output_path):
        """Visualize bubble detections on the image"""
        vis_image = image.copy()
        
        # Draw original detections in blue
        for box in bubbles:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw merged bubbles in red
        for box in merged_bubbles:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        cv2.imwrite(output_path, vis_image)

    def find_individual_bubbles(self, mask):
        """Analyze mask to find individual bubbles within conjoined bubbles"""
        # First, find all contours, including internal ones
        contours, hierarchy = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_TREE,  # Changed from EXTERNAL to TREE to get internal contours
            cv2.CHAIN_APPROX_TC89_L1
        )
        
        # Set minimum area threshold (adjust based on your images)
        self.contour_area_threshold = 100
        
        # First, find all valid contours
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.contour_area_threshold:
                # Smooth contour while preserving shape
                epsilon = 0.001 * cv2.arcLength(contour, True)
                smooth_contour = cv2.approxPolyDP(contour, epsilon, True)

                # Get rotated rectangle for better orientation analysis
                rect = cv2.minAreaRect(smooth_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(smooth_contour)
                center = (x + w/2, y + h/2)
                
                valid_contours.append({
                    'contour': smooth_contour,
                    'bbox': [x, y, x + w, y + h],
                    'center': center,
                    'area': area,
                    'rotated_box': box
                })
        
        # Sort contours by x-coordinate
        valid_contours.sort(key=lambda x: x['center'][0])
        
        # If we have multiple contours, analyze their relationships
        processed_contours = []
        if len(valid_contours) > 1:
            for i, cont1 in enumerate(valid_contours):
                is_part_of_processed = False
                
                for processed in processed_contours:
                    if np.array_equal(cont1['contour'], processed['contour']):
                        is_part_of_processed = True
                        break
                
                if is_part_of_processed:
                    continue
                    
                current_group = [cont1]
                
                # Check against other contours
                for j, cont2 in enumerate(valid_contours[i+1:], i+1):
                    # Create masks for precise overlap detection
                    mask1 = np.zeros_like(mask, dtype=np.uint8)
                    mask2 = np.zeros_like(mask, dtype=np.uint8)
                    
                    cv2.drawContours(mask1, [cont1['contour']], -1, 1, -1)
                    cv2.drawContours(mask2, [cont2['contour']], -1, 1, -1)
                    
                    # Create slightly dilated version for connection detection
                    kernel = np.ones((3,3), np.uint8)
                    mask1_dilated = cv2.dilate(mask1, kernel, iterations=1)
                    mask2_dilated = cv2.dilate(mask2, kernel, iterations=1)
                    
                    # Check for connection/overlap
                    connection = np.logical_and(mask1_dilated, mask2_dilated)
                    if np.any(connection):
                        # Calculate connection strength
                        connection_area = np.sum(connection)
                        min_perimeter = min(cv2.arcLength(cont1['contour'], True),
                                        cv2.arcLength(cont2['contour'], True))
                        
                        # If connection is significant
                        if connection_area > min_perimeter * 0.1:  # Adjust threshold as needed
                            current_group.append(cont2)
                
                # Process the group
                if len(current_group) > 1:
                    # Sort group by x-coordinate
                    current_group.sort(key=lambda x: x['center'][0])
                    
                    # Mark as conjoined and assign positions
                    for idx, cont in enumerate(current_group):
                        cont['is_conjoined'] = True
                        cont['position'] = 'left' if idx == 0 else 'right'
                        processed_contours.append(cont)
                else:
                    cont1['is_conjoined'] = False
                    cont1['position'] = None
                    processed_contours.append(cont1)
        else:
            # Single contour case
            if valid_contours:
                valid_contours[0]['is_conjoined'] = False
                valid_contours[0]['position'] = None
                processed_contours = valid_contours
        
        # Debug visualization
        debug_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for idx, cont in enumerate(processed_contours):
            color = (0, 255, 0) if not cont.get('is_conjoined') else (
                (255, 0, 0) if cont['position'] == 'left' else (0, 0, 255)
            )
            cv2.drawContours(debug_img, [cont['contour']], -1, color, -1)
            cv2.putText(debug_img, str(idx), (int(cont['center'][0]), int(cont['center'][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite('debug_contours.png', debug_img)
        
        return processed_contours
    
    def find_matching_translation_for_bubble(self, bubble, translations):
        """Find the best matching translation based on position and overlap"""
        closest_translation = None
        best_score = float('-inf')
        
        bubble_center = bubble['center']
        bubble_box = bubble['bbox']
        is_conjoined = bubble.get('is_conjoined', False)
        bubble_position = bubble.get('position', None)
        
        for trans in translations:
            if trans.get('used', False):
                continue
            
            bbox = trans['processing_box']
            trans_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
            
            # Calculate IoU and distances
            iou = self.calculate_iou(bubble_box, bbox)
            dx = abs(bubble_center[0] - trans_center[0])
            dy = abs(bubble_center[1] - trans_center[1])
            
            # Normalize distances
            norm_dx = dx / self.process_width
            norm_dy = dy / self.process_height
            
            # Base score starts with IoU
            score = iou
            
            # For conjoined bubbles, use strict position matching
            if is_conjoined:
                trans_position = 'left' if trans_center[0] < bubble_center[0] else 'right'
                if trans_position != bubble_position:
                    continue  # Skip if positions don't match
                
                # Add position bonus
                score += 0.5
            
            # Distance penalties
            score -= (0.3 * norm_dx + 0.1 * norm_dy)
            
            print(f"\nBubble {bubble_position if is_conjoined else 'single'}")
            print(f"Translation: {trans['content']['en']}")
            print(f"Score components - IoU: {iou:.3f}")
            print(f"Distance penalties - dx: {norm_dx:.3f}, dy: {norm_dy:.3f}")
            print(f"Final score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                closest_translation = trans
        
        if closest_translation:
            closest_translation['used'] = True
        
        return closest_translation
            

    def detect_bubbles(self, image):
        h, w, _ = image.shape
        size = (h, w) if h >= w * 5 else 1024
        results = self.bubble_detector(image, device=self.device, imgsz=size, conf=0.1)[0]

        # del self.bubble_detector
        # self.bubble_detector = None
        
        # Filter boxes to only include text_bubbles (class 0) and exclude text_free (class 1)
        mask = results.boxes.cls.cpu().numpy() == 0  # Only keep class 0 (text_bubbles)
        filtered_boxes = results.boxes.xyxy.cpu().numpy()[mask]
        
        return filtered_boxes
    
    def detect_sfx(self, image):
        """Detect sound effects (SFX) if SFX model is provided"""
        if not self.sfx_detector:
            return np.empty((0, 4))
        
        h, w, _ = image.shape
        size = (h, w) if h >= w * 5 else 1024

        results = self.sfx_detector(image, device=self.device, imgsz=size, conf=0.42)[0]
        if len(results.boxes) == 0:
            return np.empty((0, 4))

        return results.boxes.xyxy.cpu().numpy()


    @staticmethod
    def calculate_iou(rect1, rect2):
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[2], rect2[2])
        y2 = min(rect1[3], rect2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
        union_area = rect1_area + rect2_area - intersection_area

        iou = intersection_area / union_area if union_area != 0 else 0
        return iou

    def do_rectangles_overlap(self, rect1, rect2):
        iou = self.calculate_iou(rect1, rect2)
        return iou > self.overlap_threshold

    def merge_overlapping_boxes(self, boxes):
        merged_boxes = []
        remaining_boxes = boxes.tolist()

        while remaining_boxes:
            current_box = remaining_boxes.pop(0)
            merged = False

            for i, other_box in enumerate(remaining_boxes):
                if self.do_rectangles_overlap(current_box, other_box):
                    print(f"Merging boxes: {current_box} and {other_box}")
                    merged_box = self.merge_boxes(current_box, other_box)
                    remaining_boxes[i] = merged_box
                    merged = True
                    break

            if not merged:
                merged_boxes.append(current_box)

        return np.array(merged_boxes)

    @staticmethod
    def merge_boxes(box1, box2):
        return [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3])
        ]

    def generate_mask(self, image, bounding_box):
        self.sam_predictor.set_image(image)
        masks, _, _ = self.sam_predictor.predict(
            box=bounding_box[None, :],
            multimask_output=False
        )
        return masks[0]
    
    def scale_mask(self, mask, scale_factor):
        # Get the bounding box of the mask
        y, x = np.where(mask)
        top, left, bottom, right = np.min(y), np.min(x), np.max(y), np.max(x)
        
        # Calculate the center of the bounding box
        center_y, center_x = (top + bottom) // 2, (left + right) // 2
        
        # Calculate new dimensions
        height, width = bottom - top, right - left
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        
        # Resize the mask
        mask_resized = cv2.resize(mask[top:bottom, left:right].astype(np.uint8), 
                                  (new_width, new_height), 
                                  interpolation=cv2.INTER_LINEAR)
        
        # Create a new mask of the original size
        new_mask = np.zeros_like(mask)
        
        # Calculate where to place the resized mask
        new_top = max(0, center_y - new_height // 2)
        new_left = max(0, center_x - new_width // 2)
        new_bottom = min(mask.shape[0], new_top + new_height)
        new_right = min(mask.shape[1], new_left + new_width)
        
        # Place the resized mask
        new_mask[new_top:new_bottom, new_left:new_right] = mask_resized[:new_bottom-new_top, :new_right-new_left]
        
        return new_mask

    def find_matching_translation(self, bubble_box):
        """Find the closest matching translation for a bubble based on IOU"""
        bubble_tuple = tuple(bubble_box.astype(int).tolist())
        best_match = None
        best_iou = 0
        
        for bbox in self.translation_lookup:
            iou = self.calculate_iou(bubble_tuple, bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = bbox
        
        return self.translation_lookup[best_match] if best_match is not None else None
    
    def calculate_text_size_requirements(self, text, bubble_width, bubble_height, is_text_free=False):
        """Calculate the required size for text and determine if scaling is needed"""
        # Adjust font size based on image dimensions to maintain consistency
        size_factor = min(1.0, self.image_width / 2048)  # Base scaling on original image width
        
        # Start with maximum font size - scaled based on image size
        min_font = max(int(self.min_text_free_font_size * size_factor), 16) if is_text_free else max(int(self.min_font_size * size_factor), 12)
        max_font = max(int(self.max_text_free_font_size * size_factor), 20) if is_text_free else max(int(self.max_font_size * size_factor), 18)
        font_size = max_font
        scale_factor = 1.0
    
        width_factor = 0.98 if is_text_free else 0.95
        height_factor = 0.95 if is_text_free else 0.90
        
        effective_width = bubble_width * width_factor
        effective_height = bubble_height * height_factor
        
        # Ensure we have reasonable minimum dimensions to prevent extreme scaling
        min_effective_width = 50 * size_factor
        min_effective_height = 40 * size_factor
        
        effective_width = max(effective_width, min_effective_width)
        effective_height = max(effective_height, min_effective_height)
        
        while font_size >= min_font:
            # Estimate text dimensions at current font size
            chars_per_line = max(6, int(effective_width / (font_size * 0.8)))
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                word_length = len(word)
    
                if any(c in '.,!?"\'' for c in word):
                    word_length += 1
                if any(c.isupper() for c in word):
                    word_length += len([c for c in word if c.isupper()]) * 0.3
    
                if current_length + word_length <= chars_per_line:
                    current_line.append(word)
                    current_length += word_length + 1
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = word_length
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Calculate total height needed
            line_spacing = 1.5
            total_height = len(lines) * font_size * line_spacing 
    
            max_line_length = max(sum(len(word) + 1 for word in line.split()) for line in lines)
            total_width = max_line_length * font_size * 0.8
    
            for line in lines:
                caps_count = sum(1 for c in line if c.isupper())
                if caps_count > 2:
                    total_width = total_width * (1 + (caps_count * 0.05))
            
            # Check if text fits
            if total_height <= effective_height and total_width <= effective_width:
                return font_size, lines, scale_factor
            
            # If text doesn't fit with current font size, try scaling the bubble
            if font_size == min_font:
                height_scale = total_height / effective_height
                width_scale = total_width / effective_width
                max_scale = self.max_scale_factor * size_factor  # Scale the max scaling factor based on image size
                scale_factor = min(max(height_scale, width_scale), max_scale)
                if scale_factor > 1:
                    return font_size, lines, scale_factor
            
            font_size -= 2
        
        return min_font, lines, scale_factor
    
    def process_text_free_areas(self, image, text_blocks, bubbles, dwg):
        """Handle text-free areas with inpainting"""
        # Create mask for text-free areas
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        for block in text_blocks:
            if block['type'] == 'text_free':  # Only process text outside bubbles
                print(block)
                x1, y1, x2, y2 = map(int, block['bounding_box'])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Convert to PIL for inpainting
        mask_pil = Image.fromarray(mask)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Inpaint text-free areas
        if np.any(mask):  # Only inpaint if there are text-free areas
            cnet_image = image_pil.copy()
            cnet_image.paste(0, (0, 0), mask_pil)
            
            for inpainted in self.inpaint_pipe(
                prompt_embeds=self.prompt_embeds[0],
                negative_prompt_embeds=self.prompt_embeds[1],
                pooled_prompt_embeds=self.prompt_embeds[2],
                negative_pooled_prompt_embeds=self.prompt_embeds[3],
                image=cnet_image,
                num_inference_steps=8,
            ):
                pass
            
            # Composite inpainted result
            inpainted = inpainted.convert('RGBA')
            if inpainted.size != image_pil.size:
                print(f"Resizing inpainted image from {inpainted.size} to {image_pil.size}")
                inpainted = inpainted.resize(image_pil.size, Image.Resampling.LANCZOS)

            inpainted.save('debug_inpainted.png')
            mask_pil.save('debug_mask.png')
            image_pil.save('debug_original.png')

            image_pil = inpainted

            # Convert back to CV2 format
            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


            del self.inpaint_pipe
            self.inpaint_pipe = None
            torch.cuda.empty_cache()
        
        # Process and add translated text for text-free areas
        text_free_group = dwg.g(id=f'text_free_areas')
        for block in text_blocks:
            if block['type'] == 'text_free':
                # Get the original text properties
                x1, y1, x2, y2 = map(int, block['bounding_box'])
                translation = block['content']['en']

                # Calculate text size and alignment
                box_width = x2 - x1
                box_height = y2 - y1
                font_size, lines, required_scale = self.calculate_text_size_requirements(
                    translation, box_width, box_height, is_text_free=True
                )

                # If the text block needs scaling, adjust accordingly
                if required_scale > 1:
                    block_mask = np.zeros(mask.shape, dtype=np.uint8)
                    cv2.rectangle(block_mask, (x1, y1), (x2, y2), 255, -1)
                    scaled_mask = self.scale_mask(block_mask > 0, required_scale)
                    contours, _ = cv2.findContours(
                        scaled_mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    if contours:
                        x, y, w, h = cv2.boundingRect(contours[0])
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        box_width, box_height = x2 - x1, y2 - y1

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Render each line of text in the block
                for idx, line in enumerate(lines):
                    y_offset = (idx - len(lines) / 2 + 0.5) * font_size * 1.2
                    text_free_group.add(dwg.text(
                        line,
                        insert=(center_x, center_y + y_offset),
                        fill=block.get('text_color', '#000000'),
                        class_='text-free',
                        # stroke='#FFFFFF',
                        # stroke_width=3,
                        style=f'font-size: {font_size}px; letter-spacing: -0.03em;'
                    ))
        # dwg.add(text_free_group)

        return image, text_free_group
    
    
    def feather_mask(self, mask, feather_amount=5, expansion_iterations=10):
        """
        Feather the edges of a binary mask using dilation and blur
        
        Args:
            mask: Binary mask array
            feather_amount: Amount of feathering to apply
        
        Returns:
            Feathered mask array
        """
        # Convert to uint8 if not already
        mask_uint8 = mask.astype(np.uint8) * 255

        
        # First dilate the mask to catch edges
        kernel = np.ones((3,3), np.uint8)
        dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=expansion_iterations)
        
        # Apply gaussian blur to feather the edges
        # blurred_mask = cv2.GaussianBlur(dilated_mask, (feather_amount*2+1, feather_amount*2+1), 0)
        
        # Normalize back to binary
        # feathered_mask = blurred_mask > 127
        
        return dilated_mask
 

    def inpaint_image(self, image_pil, mask_pil):
        """
        Inpaint an image using the pipeline with proper dimension handling and size limiting
        
        Args:
            image_pil: PIL image to inpaint
            mask_pil: PIL mask where white areas will be inpainted
        
        Returns:
            Inpainted PIL image
        """
        # Get original dimensions
        original_width, original_height = image_pil.size
        
        # Set maximum dimension threshold for processing (to avoid VRAM issues)
        MAX_DIMENSION = 2048
        
        # First step: Check if any dimension exceeds MAX_DIMENSION and resize if needed
        needs_initial_resize = max(original_width, original_height) > MAX_DIMENSION
        if needs_initial_resize:
            # Calculate new dimensions while preserving aspect ratio
            scale_factor = MAX_DIMENSION / max(original_width, original_height)
            intermediate_width = int(original_width * scale_factor)
            intermediate_height = int(original_height * scale_factor)
            
            print(f"Image too large - Resizing from {original_width}x{original_height} to {intermediate_width}x{intermediate_height}")
            
            # Apply initial resize
            image_resized = image_pil.resize((intermediate_width, intermediate_height), Image.LANCZOS)
            mask_resized = mask_pil.resize((intermediate_width, intermediate_height), Image.NEAREST)
        else:
            # Skip initial resize, use original dimensions
            intermediate_width, intermediate_height = original_width, original_height
            image_resized = image_pil
            mask_resized = mask_pil
        
        # Second step: Adjust to multiples of 64 for optimal model performance
        new_width = ((intermediate_width + 63) // 64) * 64
        new_height = ((intermediate_height + 63) // 64) * 64
        
        # Resize to model-friendly dimensions if needed
        if intermediate_width != new_width or intermediate_height != new_height:
            print(f"Adjusting to multiples of 64: {intermediate_width}x{intermediate_height} to {new_width}x{new_height}")
            image_resized = image_resized.resize((new_width, new_height), Image.LANCZOS)
            # Use NEAREST for mask resizing to preserve binary nature
            mask_resized = mask_resized.resize((new_width, new_height), Image.NEAREST)
        
        # Prepare for inpainting
        cnet_image = image_resized.copy()
        cnet_image.paste(0, (0, 0), mask_resized)
        
        # Debug save
        cnet_image.save("cnet_image_debug.png")
        
        # Process with pipeline
        for inpainted in self.inpaint_pipe(
            prompt_embeds=self.prompt_embeds[0],
            negative_prompt_embeds=self.prompt_embeds[1],
            pooled_prompt_embeds=self.prompt_embeds[2],
            negative_pooled_prompt_embeds=self.prompt_embeds[3],
            image=cnet_image,
            num_inference_steps=8,
        ):
            final_result = inpainted
        
        # Convert to RGBA for transparency support
        final_result = final_result.convert("RGBA")

        final_result.save("final_result.png")
        
        # Composite with resized image
        result = image_resized.copy()
        result.paste(final_result, (0, 0), mask_resized)
        
        # Resize back to original dimensions if any resizing was done
        if original_width != new_width or original_height != new_height:
            print(f"Restoring to original size: {new_width}x{new_height} to {original_width}x{original_height}")
            result = result.resize((original_width, original_height), Image.LANCZOS)
        
        return result
    
    def process_page(self, image_path, output_folder):

        job_id = os.path.basename(output_folder)
        image_basename = os.path.splitext(os.path.basename(image_path))[0]

        os.makedirs(output_folder, exist_ok=True)

        output_kra_path = os.path.join(output_folder, f"{image_basename}.kra")
        output_image_path = os.path.join(output_folder, f"{image_basename}.png")

        MAX_DIMENSION = 2048

        # Read the image
        image = cv2.imread(image_path)

        # Store original dimensions
        original_height, original_width = image.shape[:2]
        self.image_width, self.image_height = original_width, original_height

        # Calculate a base sizing factor for text relative to image size
        # This ensures consistent text sizes regardless of initial image dimensions
        self.base_size_factor = min(1.0, original_width / 2048)
        
        # Check if image is too large and needs initial resizing
        needs_initial_resize = max(original_width, original_height) > MAX_DIMENSION
        
        if needs_initial_resize:
            # Calculate scale factor to keep aspect ratio
            scale_factor = MAX_DIMENSION / max(original_width, original_height)
            process_width = int(original_width * scale_factor)
            process_height = int(original_height * scale_factor)
            
            print(f"Image too large - Resizing from {original_width}x{original_height} to {process_width}x{process_height}")
            
            # Resize for processing (all operations will be done on this smaller size)
            image = cv2.resize(image, (process_width, process_height), interpolation=cv2.INTER_AREA)
        else:
            process_width, process_height = original_width, original_height
            scale_factor = 1.0

        # Store processing dimensions for other operations
        self.process_width = process_width
        self.process_height = process_height
        self.scale_factor = scale_factor
        self.scale_back_factor = 1.0 / scale_factor if needs_initial_resize else 1.0
        
        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect bubbles
        bubbles = self.detect_bubbles(image)
        print("Done detecting bubbles")
        merged_bubbles = self.merge_overlapping_boxes(bubbles)
        print("Done merging bubbles")
        sfx_boxes = self.detect_sfx(image)
        print("Done detecting sfx")

        # self.setup_inpainting()

        kra_doc = KritaDocument(width=original_width, height=original_height)


        # Scale translation data to match processing size
        processed_translation_data = []
        for block in self.translation_data:
            # Create a copy to avoid modifying the original
            processed_block = block.copy()
            
            # Scale the bounding box to match the processing size
            original_box = block['bounding_box']
            scaled_box = [
                int(original_box[0] * scale_factor),
                int(original_box[1] * scale_factor),
                int(original_box[2] * scale_factor),
                int(original_box[3] * scale_factor)
            ]
            processed_block['processing_box'] = scaled_box
            processed_translation_data.append(processed_block)
        
        # Use the processed translation data for bubble matching
        self.processed_translation_data = processed_translation_data
    
        # Process text blocks and identify text-free areas
        text_blocks = []
        for block in processed_translation_data:
            # Check for overlap with bubbles (using processing_box)
            is_in_bubble = False
            for bubble in merged_bubbles:
                if self.calculate_iou(block['processing_box'], bubble) > self.overlap_threshold:
                    is_in_bubble = True
                    break
            block['type'] = 'text_bubble' if is_in_bubble else 'text_free'
            text_blocks.append(block)

        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Add text-free areas to mask
        for block in text_blocks:
            if block['type'] == 'text_free':
                x1, y1, x2, y2 = map(int, block['processing_box'])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Add SFX areas to mask
        for box in sfx_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        
        # Add bubble areas to mask
        for bubble in merged_bubbles:
            x1, y1, x2, y2 = map(int, bubble)
            bubble_mask = self.generate_mask(image_rgb, bubble)

            kernel = np.ones((3, 3), np.uint8)
            dilated_bubble_mask = cv2.dilate(bubble_mask.astype(np.uint8) * 255, kernel, iterations=2)

            mask = np.logical_or(mask, dilated_bubble_mask > 0).astype(np.uint8) * 255

        # alpha_channel = mask_rgba.split()[3]
        # binary_mask = alpha_channel.point(lambda p: p > 0 and 255)

        # Save debug masks
        cv2.imwrite('debug_mask_final.png', mask)
        # Convert to PIL for inpainting
        mask_pil = Image.fromarray(mask)
        image_pil = Image.fromarray(image_rgb)

        mask_pil.save('debug_combined_mask.png')
        

        # Perform inpainting if there are areas to inpaint
        if np.any(mask):

            print("Inpainting manga page...")
            inpainted_image = self.inpaint_image(image_pil, mask_pil)
            inpainted_image.save(output_image_path)

            image = cv2.cvtColor(np.array(inpainted_image), cv2.COLOR_RGBA2RGB)

            # Clean up to free memory
            del self.inpaint_pipe
            self.inpaint_pipe = None
            torch.cuda.empty_cache()

        if needs_initial_resize:
            inpainted_full_size = cv2.resize(image, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            inpainted_full_size = image

        inpainted_pil = Image.fromarray(cv2.cvtColor(inpainted_full_size, cv2.COLOR_RGBA2RGB))
        original_pil = Image.open(image_path)



        text_free_font_multiplier = 1.4 * self.base_size_factor

        # Render text-free areas
        for block in text_blocks:
            if block['type'] == 'text_free':
                x1, y1, x2, y2 = map(int, [
                    block['bounding_box'][0], 
                    block['bounding_box'][1],
                    block['bounding_box'][2],
                    block['bounding_box'][3]
                ])
                translation = block['content']['en']
                
                box_width = x2 - x1
                box_height = y2 - y1
                font_size, lines, required_scale = self.calculate_text_size_requirements(
                    translation, box_width, box_height, is_text_free=True
                )

                # font_size = int(font_size * text_free_font_multiplier)

                text_style = TextStyle(
                    font_family="Anime Ace 2.0 BB",
                    font_size=font_size,
                    fill_color=block.get('text_color', '#000000'),
                    text_align="center",
                    text_align_last="center",
                    line_height=1.2,
                    word_spacing=-1,
                    letter_spacing=0,
                    text_anchor="middle",
                )
                
                # Create text layer for text-free area
                text_layer = ShapeLayer.from_text(
                    "\n".join(lines),
                    style=text_style,
                    x=(x1 + x2) / 2,
                    y=(y1 + y2) / 2,
                    layer_style=LayerStyle(  # Add layer style
                        enabled=True,
                        stroke_enabled=True,
                        stroke_color=(255, 255, 255),
                        stroke_size=5.0 * self.base_size_factor,
                        stroke_opacity=100.0
                    )
                )
                kra_doc.add_text_layer(text_layer, name=f"TextFree_{x1}_{y1}")


        # scale_back_factor = 1.0 / scale_factor if needs_initial_resize else 1.0

        for i, merged_bubble in enumerate(merged_bubbles):
            # Generate mask for the merged bubble
            mask = self.generate_mask(image_rgb, merged_bubble)
            
            # Find individual bubbles within the merged bubble
            individual_bubbles = self.find_individual_bubbles(mask)
            
            # Draw bubble outlines and process text
            for j, bubble in enumerate(individual_bubbles):
                
                # Scale contour back to original size if needed
                if needs_initial_resize:
                    # Scale contour points
                    original_contour = bubble['contour'].copy()
                    for point_idx in range(len(original_contour)):
                        original_contour[point_idx][0][0] = int(original_contour[point_idx][0][0] * self.scale_back_factor)
                        original_contour[point_idx][0][1] = int(original_contour[point_idx][0][1] * self.scale_back_factor)
                    
                    # Scale bounding box and center
                    original_bbox = [
                        int(bubble['bbox'][0] * self.scale_back_factor),
                        int(bubble['bbox'][1] * self.scale_back_factor),
                        int(bubble['bbox'][2] * self.scale_back_factor),
                        int(bubble['bbox'][3] * self.scale_back_factor)
                    ]
                    original_center = (
                        bubble['center'][0] * self.scale_back_factor,
                        bubble['center'][1] * self.scale_back_factor
                    )
                    
                    # Use original size for Krita document
                    contour = original_contour
                    bubble_bbox = original_bbox
                    bubble_center = original_center
                else:
                    contour = bubble['contour']
                    bubble_bbox = bubble['bbox']
                    bubble_center = bubble['center']

                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                path_data = "M"
                for point in approx.reshape(-1, 2):
                    x, y = point
                    x_pt = x
                    y_pt = y
                    path_data += f"{x_pt:.2f},{y_pt:.2f} L"
                path_data = path_data[:-2] + "Z"
                
                # Create bubble path with style
                bubble_style = ShapeStyle(
                    fill="white",
                    fill_opacity=1,
                    stroke="#000000",
                    stroke_width=2.5
                )

                # Create a copy of the bubble for matching purposes
                matching_bubble = bubble.copy()
                
                # If we need matching with processed translation coordinates, convert center and bbox
                if needs_initial_resize:
                    matching_bubble['center'] = (
                        bubble['center'][0],  # keep as-is since we're matching in processed coordinates
                        bubble['center'][1]
                    )
                    matching_bubble['bbox'] = bubble['bbox']  # keep as-is
                
                # Find and process text
                translation_info = self.find_matching_translation_for_bubble(
                    bubble,
                    [t for t in self.processed_translation_data if not t.get('used', False)]
                )
                
                if translation_info:
                    print(f"Matched text: {translation_info['content']['en']} to bubble {i}_{j}")
                    
                    # Calculate text requirements
                    bubble_width = bubble_bbox[2] - bubble_bbox[0]
                    bubble_height = bubble_bbox[3] - bubble_bbox[1]
                    
                    font_size, lines, required_scale = self.calculate_text_size_requirements(
                        translation_info['content']['en'],
                        bubble_width,
                        bubble_height
                    )

                    # Calculate total text block height
                    line_height = font_size * 1.2
                    total_text_height = len(lines) * line_height
                    
                    # Calculate starting Y position to center entire text block
                    center_y = bubble_center[1] - (total_text_height / 2) + (line_height / 2)
                    

                    text_style = TextStyle(
                    font_family="Anime Ace 2.0 BB",
                    font_size=font_size,
                    fill_color=translation_info['text_color'],
                    text_align="center",
                    text_align_last="center",
                    line_height=1.4,
                    word_spacing=0,
                    letter_spacing=0,
                    text_anchor="middle",
                    )
                    
                    # Create text layer for each bubble
                    text_layer = ShapeLayer.from_text(
                        "\n".join(lines),
                        style=text_style,
                        x=bubble_center[0],
                        y=center_y,
                        layer_style=LayerStyle(  # Add layer style
                            stroke_enabled=True,
                            stroke_size=5.0 * self.base_size_factor,
                            stroke_color=(255, 255, 255),  # White stroke (or user can use hex code)
                            stroke_opacity=100.0
                        )
                    )
                    kra_doc.add_text_layer(text_layer, name=f"BubbleText_{i}_{j}")

                bubble_path = Path(d=path_data, style=bubble_style)
                
                # Add bubble as shape layer
                kra_doc.add_shape_layer(
                    bubble_path,
                    name=f"Bubble_{i}_{j}",
                    opacity=255
                )

        # Add inpainted background
        kra_doc.add_image_layer(inpainted_pil, name="Inpainted Background", opacity=255)
                    
        # Add original image
        kra_doc.add_image_layer(original_pil, name="Original Image", opacity=255)

        kra_doc.save(output_kra_path)
        print(f"Saved Krita document to: {output_kra_path}")

    


    def save_output_files(self, svg_path, background_path, width, height):
        """
        Save both SVG and PNG versions of the output, compositing with background
        
        Args:
            svg_path: Path to the SVG file
            background_path: Path to the background image
            width: Image width
            height: Image height
        """
        import cairosvg
        from PIL import Image
        
        # Generate PNG path from SVG path
        png_path = svg_path.rsplit('.', 1)[0] + '.png'
        
        try:
            # First convert SVG to PNG with transparency
            temp_png_path = "temp_overlay.png"

            with open(svg_path, 'r') as file:
                svg_content = file.read()
            
            # Remember: Made an update to cairosvg codebase (surface.py) to fix the issue with text rendering with paint-order="stroke"
            cairosvg.svg2png(
                bytestring=svg_content,
                write_to=temp_png_path,
                output_width=width,
                output_height=height,
                unsafe=True
            )
            
            # Open both images
            background = Image.open(background_path)
            overlay = Image.open(temp_png_path)
            
            # Convert background to RGBA if it isn't already
            if background.mode != 'RGBA':
                background = background.convert('RGBA')
            
            # Ensure overlay is RGBA
            if overlay.mode != 'RGBA':
                overlay = overlay.convert('RGBA')
                
            # Composite the images
            final_image = Image.alpha_composite(background, overlay)
            
            # Save the result
            final_image.save(png_path, 'PNG')
            
            # Clean up temporary file
            import os
            os.remove(temp_png_path)
            
            print(f"Complete PNG saved to: {png_path}")
        except Exception as e:
            print(f"Error creating final PNG: {e}")


    def process_with_translations(self, image_path: str, translation_data: list, output_kra_path: str):
        """Process page with direct translation data"""
        self.translation_data = translation_data  # Store full translation data
        self.process_page(image_path, output_kra_path)

    def get_model_path(self, component_repo: str, model_name: str) -> str:
        storage_folder = os.path.join(
            self.cache_dir, "models--" + component_repo.replace("/", "--")
        )
        if not os.path.exists(storage_folder):
            raise FileNotFoundError(f"Model {component_repo} not found")
        refs_path = os.path.join(storage_folder, "refs", "main")
        if not os.path.exists(refs_path):
            raise FileNotFoundError(f"No commit hash found")
        with open(refs_path, "r") as f:
            commit_hash = f.read().strip()
        checkpoint = os.path.join(
            storage_folder, "snapshots", commit_hash, model_name
        )
        return checkpoint
    

# Usage
def main(image_path: str, output_kra_path: str, target_languages: list[str] = ["en", "zh", "es"]):
    # Initialize both components
    bubble_model_path = 'models/comic-speech-bubble-detector.pt'
    translator = MangaTranslator(bubble_model_path)
    processor = MangaPageProcessor(bubble_model_path)
    
    # Process image through the pipeline
    print("Detecting and translating text...")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if os.path.exists(f'translation_result_{base_name}.json'):
        with open(f'translation_result_{base_name}.json', 'r', encoding='utf-8') as f:
            translation_data = json.load(f)
    else:
        image, translation_data = translator.process_image(image_path, target_languages)
    
    print("Generating SVG with translations...")
    processor.process_with_translations(image_path, translation_data, output_kra_path)
    
    print(f"Processing complete. KRA saved to: {output_kra_path}")

def process_manga_folder(input_folder, output_folder, target_languages=["en", "zh", "es"]):
        """
        Process all manga images in a folder
        
        Args:
            input_folder: Path to folder containing manga images
            output_folder: Path to save output KRA files
            target_languages: List of target languages for translation
        """
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize components
        bubble_model_path = 'models/comic-speech-bubble-detector.pt'
        translator = MangaTranslator(bubble_model_path)
        processor = MangaPageProcessor(bubble_model_path)
        
        # Get all image files in the input folder
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        image_files_set = set()
        for ext in image_extensions:
            image_files_set.update(glob.glob(os.path.join(input_folder, f'*{ext}')))
            image_files_set.update(glob.glob(os.path.join(input_folder, f'*{ext.upper()}')))

        image_files = sorted(list(image_files_set))

        print(image_files)
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for image_path in image_files:
            try:
                # Get base filename without extension
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                print(f"Processing {base_name}...")
                
                # Define output paths
                output_kra_path = os.path.join(output_folder, f"{base_name}.kra")
                translation_json_path = os.path.join(output_folder, f"{base_name}_translation.json")
                
                # Check if translation already exists
                if os.path.exists(translation_json_path):
                    print(f"Loading existing translation from {translation_json_path}")
                    with open(translation_json_path, 'r', encoding='utf-8') as f:
                        translation_data = json.load(f)
                else:
                    # Generate translations
                    print(f"Generating translations for {base_name}...")
                    image, translation_data = translator.process_image_with_custom_output(
                        image_path, 
                        target_languages,
                        translation_json_path
                    )
                
                # Process with translations
                print(f"Creating Krita document for {base_name}...")
                processor.process_with_translations(image_path, translation_data, output_kra_path)
                
                print(f"Successfully processed {base_name}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        print(f"Batch processing complete! Processed {len(image_files)} images.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # Batch process a folder
        if len(sys.argv) < 4:
            print("Usage for batch processing: python script.py --batch input_folder output_folder")
            sys.exit(1)
        
        input_folder = sys.argv[2]
        output_folder = sys.argv[3]
        process_manga_folder(input_folder, output_folder)
    else:
        # Process a single image (original behavior)
        image_path = "004.jpg"
        output_kra_path = "kra_files/output_manga_page_004.kra"
        main(image_path, output_kra_path)