import torch
import glob
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub import hf_hub_download
from diffusers.models.model_loading_utils import load_state_dict
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers import AutoencoderKL
from controlnet_union import ControlNetModel_Union




class MangaCleaner:
    def __init__(self, bubble_model_path, sfx_model_path=None, overlap_threshold=0.05):
        self.cache_dir = HF_HUB_CACHE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.overlap_threshold = overlap_threshold
        self.contour_area_threshold = 100
        self.debug = False
        self.debug_dir = "debug"

        self._load_models(bubble_model_path, sfx_model_path)
        self._setup_inpainting()

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

    def _setup_inpainting(self):
        """Setup inpainting pipeline for cleaning text areas"""
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

        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=dtype
        ).to(self.device)

        # Initialize pipeline
        self.inpaint_pipe = StableDiffusionXLFillPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0",
            torch_dtype=dtype,
            vae=vae,
            controlnet=model,
            variant="fp16",
        ).to(self.device, torch.float16)

        self.inpaint_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.inpaint_pipe.scheduler.config
        )

        # Pre-encode prompt for inpainting
        prompt = "high quality"
        self.prompt_embeds = self.inpaint_pipe.encode_prompt(prompt, self.device, True)

    def load_sam(self, checkpoint_path):
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        return SamPredictor(sam)

    def detect_bubbles_and_text(self, image):
        h, w, _ = image.shape
        size = (h, w) if h >= w * 5 else 1024
        
        # Detect bubbles and free text with the bubble detector
        results = self.bubble_detector(image, device=self.device, imgsz=size, conf=0.1)[0]
        
        # Get all boxes - both text bubbles (class 0) and free text (class 1)
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        
        # Separate bubbles and free text
        bubble_boxes = boxes[classes == 0]
        free_text_boxes = boxes[classes == 1]
        
        # Detect SFX with the SFX detector if available
        sfx_boxes = np.empty((0, 4))
        if hasattr(self, 'sfx_detector') and self.sfx_detector is not None:
            sfx_results = self.sfx_detector(image, device=self.device, imgsz=size, conf=0.42)[0]
            if len(sfx_results.boxes) > 0:
                sfx_boxes = sfx_results.boxes.xyxy.cpu().numpy()
                print(f"Detected {len(sfx_boxes)} SFX elements")
        
        return bubble_boxes, free_text_boxes, sfx_boxes

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
        if len(boxes) == 0:
            return np.array([])
            
        merged_boxes = []
        remaining_boxes = boxes.tolist()

        while remaining_boxes:
            current_box = remaining_boxes.pop(0)
            merged = False

            for i, other_box in enumerate(remaining_boxes):
                if self.do_rectangles_overlap(current_box, other_box):
                    if self.debug:
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
    
    def feather_mask(self, mask, expansion_iterations=10):
        """
        Feather the edges of a binary mask using dilation
        
        Args:
            mask: Binary mask array
            expansion_iterations: Number of dilation iterations
        
        Returns:
            Dilated mask array
        """
        # Convert to uint8 if not already
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # First dilate the mask to catch edges
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=expansion_iterations)
        
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
        if self.debug:
            cnet_image.save(os.path.join(self.debug_dir, "cnet_image_debug.png"))
        
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
        
        if self.debug:
            final_result.save(os.path.join(self.debug_dir, "final_result_before_resize.png"))
        
        # Composite with resized image
        # result = image_resized.copy()
        # result.paste(final_result, (0, 0), mask_resized)
        
        # Resize back to original dimensions if any resizing was done
        if original_width != new_width or original_height != new_height:
            print(f"Restoring to original size: {new_width}x{new_height} to {original_width}x{original_height}")
            result = final_result.resize((original_width, original_height), Image.LANCZOS)
        
        return result

    def clean_page(self, image_path, output_path, debug=False, debug_dir="debug"):
        """
        Clean text and bubbles from a manga page
        
        Args:
            image_path: Path to the manga image
            output_path: Path to save the cleaned image
            debug: Whether to save debug images
            debug_dir: Directory to save debug images
        
        Returns:
            True if cleaning was successful, False otherwise
        """
        self.debug = debug
        self.debug_dir = debug_dir
        
        if debug:
            os.makedirs(debug_dir, exist_ok=True)

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Store original dimensions
        original_height, original_width = image.shape[:2]
        
        # Check if image is too large and needs initial resizing for detection
        MAX_DIMENSION = 2048
        needs_initial_resize = max(original_width, original_height) > MAX_DIMENSION
        
        if needs_initial_resize:
            # Calculate scale factor to keep aspect ratio
            scale_factor = MAX_DIMENSION / max(original_width, original_height)
            process_width = int(original_width * scale_factor)
            process_height = int(original_height * scale_factor)
            
            print(f"Image too large - Resizing from {original_width}x{original_height} to {process_width}x{process_height} for detection")
            
            # Resize for processing (all operations will be done on this smaller size)
            processing_image = cv2.resize(image, (process_width, process_height), interpolation=cv2.INTER_AREA)
        else:
            processing_image = image.copy()
            
        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(processing_image, cv2.COLOR_BGR2RGB)
        
        # Initialize inpainting
        # self.setup_inpainting()
        
        # Detect bubbles, free text, and SFX
        bubble_boxes, free_text_boxes, sfx_boxes = self.detect_bubbles_and_text(processing_image)
        merged_bubbles = self.merge_overlapping_boxes(bubble_boxes)
        
        # Create combined mask
        mask = np.zeros((processing_image.shape[0], processing_image.shape[1]), dtype=np.uint8)
        
        # Add text-free (outside bubbles) areas to mask
        for box in free_text_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            
        # Add SFX areas to mask
        for box in sfx_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Add bubble areas to mask using SAM for precise segmentation
        for bubble in merged_bubbles:
            x1, y1, x2, y2 = map(int, bubble)
            bubble_mask = self.generate_mask(image_rgb, bubble)
            
            # Dilate the mask slightly to ensure complete text removal
            dilated_bubble_mask = self.feather_mask(bubble_mask, expansion_iterations=2)
            
            # Combine with main mask
            mask = np.logical_or(mask, dilated_bubble_mask > 0).astype(np.uint8) * 255
        
        # Save debug mask if enabled
        if debug:
            cv2.imwrite(os.path.join(debug_dir, 'debug_mask_final.png'), mask)
        
        # Convert to PIL for inpainting
        mask_pil = Image.fromarray(mask)
        image_pil = Image.fromarray(image_rgb)
        
        if debug:
            mask_pil.save(os.path.join(debug_dir, 'debug_mask.png'))
            image_pil.save(os.path.join(debug_dir, 'debug_original.png'))
        
        # Perform inpainting if there are areas to inpaint
        if np.any(mask):
            print("Inpainting manga page...")
            inpainted_image = self.inpaint_image(image_pil, mask_pil)
            
            if debug:
                inpainted_image.save(os.path.join(debug_dir, 'debug_inpainted.png'))
            
            # If we resized the image for processing, resize the result back to original dimensions
            if needs_initial_resize:
                print(f"Restoring inpainted image to original size {original_width}x{original_height}")
                inpainted_image = inpainted_image.resize((original_width, original_height), Image.LANCZOS)
                
                if debug:
                    inpainted_image.save(os.path.join(debug_dir, 'debug_inpainted_original_size.png'))
            
            # Save the final inpainted image
            inpainted_image.save(output_path)
            print(f"Cleaned image saved to: {output_path}")
            return True
        else:
            print("No text or bubbles detected. Saving original image.")
            # If we used a processed image, convert back to original size
            if needs_initial_resize:
                original_image = Image.open(image_path)
                original_image.save(output_path)
            else:
                image_pil.save(output_path)
            return True

    def clean_pages_batch(self, input_folder, output_folder, debug=False, debug_dir="debug"):
        """
        Clean text and bubbles from a batch of manga pages
        
        Args:
            input_folder: Folder containing manga images
            output_folder: Folder to save cleaned images
            debug: Whether to save debug images
            debug_dir: Directory to save debug images
        """
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        if debug:
            os.makedirs(debug_dir, exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        image_files_set = set()
        for ext in image_extensions:
            image_files_set.update(glob.glob(os.path.join(input_folder, f'*{ext}')))
            image_files_set.update(glob.glob(os.path.join(input_folder, f'*{ext.upper()}')))
        
        image_files = sorted(list(image_files_set))
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        success_count = 0
        for i, image_path in enumerate(image_files):
            try:
                # Get base filename without extension
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                print(f"\nProcessing {i+1}/{len(image_files)}: {base_name}...")
                
                # Define output path
                output_path = os.path.join(output_folder, f"{base_name}_cleaned.png")
                
                # Define debug directory for this image if in debug mode
                image_debug_dir = os.path.join(debug_dir, base_name) if debug else debug_dir
                
                # Process the image
                success = self.clean_page(
                    image_path, 
                    output_path,
                    debug=debug,
                    debug_dir=image_debug_dir
                )
                
                if success:
                    success_count += 1
                    print(f"Successfully cleaned {base_name}")
                else:
                    print(f"Warning: Cleaning may not have been successful for {base_name}")
                
                # Cleanup after each image to save memory
                if hasattr(self, 'inpaint_pipe') and self.inpaint_pipe is not None:
                    # del self.inpaint_pipe
                    # self.inpaint_pipe = None
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
        
        print(f"\nBatch processing complete! Successfully processed {success_count}/{len(image_files)} images.")

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

    def cleanup(self):
        """Free GPU memory after processing"""
        if hasattr(self, 'inpaint_pipe') and self.inpaint_pipe is not None:
            del self.inpaint_pipe
            self.inpaint_pipe = None
        
        if hasattr(self, 'sam_predictor') and self.sam_predictor is not None:
            del self.sam_predictor.model
            del self.sam_predictor
            self.sam_predictor = None
            
        if hasattr(self, 'bubble_detector') and self.bubble_detector is not None:
            del self.bubble_detector
            self.bubble_detector = None
            
        if hasattr(self, 'sfx_detector') and self.sfx_detector is not None:
            del self.sfx_detector
            self.sfx_detector = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean text from manga pages")
    parser.add_argument("--input", required=True, help="Input image path or folder")
    parser.add_argument("--output", required=True, help="Output image path or folder")
    parser.add_argument("--model", default="models/comic-speech-bubble-detector.pt", help="Path to bubble detector model")
    parser.add_argument("--sfx_model", default=None, help="Path to SFX detector model (optional)")
    parser.add_argument("--batch", action="store_true", help="Process a batch of images")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    parser.add_argument("--debug_dir", default="debug", help="Directory to save debug images")
    
    args = parser.parse_args()
    
    # Initialize cleaner
    cleaner = MangaCleaner(args.model, args.sfx_model)
    
    try:
        if args.batch:
            cleaner.clean_pages_batch(args.input, args.output, args.debug, args.debug_dir)
        else:
            cleaner.clean_page(args.input, args.output, args.debug, args.debug_dir)
    finally:
        # Clean up resources
        cleaner.cleanup()