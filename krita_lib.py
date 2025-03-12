from PIL import Image
import xml.etree.ElementTree as ET
from datetime import datetime
import uuid
import os
from zipfile import ZipFile, ZIP_STORED
import math
import numpy as np
from typing import Tuple, List, Optional, Union, Dict, BinaryIO
import struct
from enum import Enum
from dataclasses import dataclass, field
import shutil
import tempfile
from xml.dom import minidom

class ByteOrder(Enum):
    BIG_ENDIAN = 1
    LITTLE_ENDIAN = 2


class ASLWriter:
    """Main class for converting XML back to ASL files."""

    def __init__(self, byte_order=ByteOrder.BIG_ENDIAN, debug=False):
        self.byte_order = byte_order
        self.format_char = '>' if byte_order == ByteOrder.BIG_ENDIAN else '<'
        self.debug = debug
        self.style_format_version = 16  # Default version

    def write_file(self, xml_path: str, asl_path: str) -> None:
        """
        Convert XML to ASL file.
        
        Args:
            xml_path: Path to the XML file
            asl_path: Path to save the ASL output
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            if root.tag != "asl":
                raise Exception("Root element must be 'asl'")
            
            with open(asl_path, 'wb') as f:
                self._write_file_impl(root, f)
                
            if self.debug:
                print(f"Successfully converted {xml_path} to {asl_path}")
            
        except Exception as e:
            print(f"Error: {e}")
            raise

    def _write_file_impl(self, root: ET.Element, device: BinaryIO) -> None:
        # Write header
        self._write_uint16(device, 2)  # styles_version
        self._write_uint32(device, 0x3842534C)  # '8BSL' signature
        self._write_uint16(device, 3)  # patterns_version
        
        # Process patterns (simplified for now)
        self._write_uint32(device, 0)  # No patterns
        
        # Find all style pairs (null descriptor + Styl descriptor)
        style_pairs = self._find_style_pairs(root)
        
        # Write number of styles
        self._write_uint32(device, len(style_pairs))
        
        # Write each style pair
        for i, (null_node, style_node) in enumerate(style_pairs):
            # Calculate and remember position for style size
            style_size_pos = device.tell()
            self._write_uint32(device, 0)  # Placeholder for style size
            
            style_start_pos = device.tell()
            
            # Write style format version
            self._write_uint32(device, self.style_format_version)
            
            # Write null descriptor
            self._write_descriptor(device, null_node)
            
            # Write style format version again
            self._write_uint32(device, self.style_format_version)
            
            # Write style descriptor
            self._write_descriptor(device, style_node)
            
            # Go back and write the actual style size
            style_end_pos = device.tell()
            style_size = style_end_pos - style_start_pos
            
            # Check if we need to pad for alignment
            padding = 0
            if style_size % 4 != 0:
                padding = 4 - (style_size % 4)
                device.write(b'\0' * padding)
                style_end_pos += padding
                style_size += padding
            
            device.seek(style_size_pos)
            self._write_uint32(device, style_size)
            device.seek(style_end_pos)  # Return to end

    def _find_style_pairs(self, root: ET.Element) -> List[Tuple[ET.Element, ET.Element]]:
        """Find pairs of null descriptor + Styl descriptor nodes."""
        style_pairs = []
        nodes = list(root.findall("./node"))
        
        # First try to find perfect pairs - null followed by Styl
        i = 0
        while i < len(nodes) - 1:
            if (nodes[i].get("classId") == "null" and 
                nodes[i+1].get("classId") == "Styl"):
                style_pairs.append((nodes[i], nodes[i+1]))
                i += 2
            else:
                i += 1
        
        return style_pairs

    def _write_descriptor(self, device: BinaryIO, node: ET.Element) -> None:
        """Write a descriptor to the ASL file."""
        name = node.get("name", "")
        class_id = node.get("classId", "")
        
        self._write_unicode_string(device, name)
        self._write_var_string(device, class_id)
        
        # Count children that need to be written
        child_nodes = list(node.findall("./node"))
        self._write_uint32(device, len(child_nodes))
        
        # Write children
        for child in child_nodes:
            self._write_child_object(device, child)

    def _write_child_object(self, device: BinaryIO, node: ET.Element, skip_key: bool = False) -> None:
        """Write a child object to the ASL file."""
        key = node.get("key", "")
        node_type = node.get("type", "")
        
        if not skip_key:
            self._write_var_string(device, key)
        
        if node_type == "Descriptor":
            class_id = node.get("classId", "")
            os_type = self._determine_os_type(node)
            self._write_fixed_string(device, os_type)
            self._write_descriptor(device, node)
        
        elif node_type == "List":
            self._write_fixed_string(device, "VlLs")
            child_nodes = list(node.findall("./node"))
            self._write_uint32(device, len(child_nodes))
            
            for child in child_nodes:
                self._write_child_object(device, child, True)
        
        elif node_type == "Double":
            self._write_fixed_string(device, "doub")
            try:
                value = float(node.get("value", "0"))
                self._write_double(device, value)
            except ValueError:
                self._write_double(device, 0.0)
        
        elif node_type == "UnitFloat":
            self._write_fixed_string(device, "UntF")
            unit = node.get("unit", "#Prc")
            try:
                value = float(node.get("value", "0"))
                self._write_fixed_string(device, unit)
                self._write_double(device, value)
            except ValueError:
                self._write_fixed_string(device, unit)
                self._write_double(device, 0.0)
        
        elif node_type == "Text":
            self._write_fixed_string(device, "TEXT")
            value = node.get("value", "")
            self._write_unicode_string(device, value)
        
        elif node_type == "Enum":
            self._write_fixed_string(device, "enum")
            type_id = node.get("typeId", "")
            value = node.get("value", "")
            
            self._write_var_string(device, type_id)
            self._write_var_string(device, value)
        
        elif node_type == "Integer":
            self._write_fixed_string(device, "long")
            try:
                value = int(node.get("value", "0"))
                self._write_uint32(device, value)
            except ValueError:
                self._write_uint32(device, 0)
        
        elif node_type == "Boolean":
            self._write_fixed_string(device, "bool")
            try:
                value = int(node.get("value", "0"))
                self._write_uint8(device, value)
            except ValueError:
                self._write_uint8(device, 0)

    def _determine_os_type(self, node: ET.Element) -> str:
        """Determine the appropriate OSType for a descriptor."""
        class_id = node.get("classId", "")
        key = node.get("key", "")
        
        # Special cases based on patterns observed in ASL files
        if class_id == "RGBC" and key == "Clr ":
            return "Objc"
        else:
            # Default is "Objc" for most descriptors
            return "Objc"

    # Basic data writing methods
    def _write_uint8(self, device: BinaryIO, value: int) -> None:
        device.write(struct.pack(f"{self.format_char}B", value))
    
    def _write_uint16(self, device: BinaryIO, value: int) -> None:
        device.write(struct.pack(f"{self.format_char}H", value))
    
    def _write_uint32(self, device: BinaryIO, value: int) -> None:
        device.write(struct.pack(f"{self.format_char}I", value))
    
    def _write_double(self, device: BinaryIO, value: float) -> None:
        device.write(struct.pack(f"{self.format_char}d", value))
    
    def _write_fixed_string(self, device: BinaryIO, value: str) -> None:
        # Pad or truncate to exactly 4 bytes
        data = value.encode('ascii', errors='replace')
        if len(data) < 4:
            data = data + b'\0' * (4 - len(data))
        elif len(data) > 4:
            data = data[:4]
        device.write(data)
    
    def _write_var_string(self, device: BinaryIO, value: str) -> None:
        data = value.encode('ascii', errors='replace')
        if not data:
            # Empty strings are written as length 4, all zeros
            self._write_uint32(device, 4)
            device.write(b'\0\0\0\0')
        else:
            self._write_uint32(device, len(data))
            device.write(data)
    
    def _write_pascal_string(self, device: BinaryIO, value: str) -> None:
        data = value.encode('ascii', errors='replace')
        if len(data) > 255:
            data = data[:255]
        self._write_uint8(device, len(data))
        device.write(data)
    
    def _write_unicode_string(self, device: BinaryIO, value: str) -> None:
        # Unicode strings in ASL are null-terminated
        if not value:
            self._write_uint32(device, 0)
            return
        
        # Convert to UTF-16BE or UTF-16LE based on byte order
        encoding = 'utf-16be' if self.byte_order == ByteOrder.BIG_ENDIAN else 'utf-16le'
        
        # Ensure null termination
        if not value.endswith('\0'):
            value += '\0'
            
        data = value.encode(encoding)
        
        # Count characters (not bytes)
        char_count = len(value)
        
        # Write character count and data
        self._write_uint32(device, char_count)
        device.write(data)


@dataclass
class LayerStyle:
    """
    Represents a layer style with effects like stroke, shadow, etc.
    Currently supports stroke effect.
    """
    # Master switch for all effects
    enabled: bool = True
    
    # Stroke effect
    stroke_enabled: bool = False
    stroke_position: str = "OutF"  # OutF (outside), InsF (inside), CtrF (center)
    stroke_blend_mode: str = "Nrml"  # Normal blend mode
    stroke_size: float = 3.0  # Size in pixels
    stroke_opacity: float = 100.0  # Opacity percentage (0-100)
    stroke_color: Union[Tuple[int, int, int], str] = (255, 255, 255)  # RGB tuple or hex color
    
    # Other effects can be added here in the future
    
    def __post_init__(self):
        # Generate a unique ID for this layer style
        self.uuid = str(uuid.uuid4())
    
    def generate_xml(self, layer_name: str) -> ET.Element:
        """
        Generate XML representation of this layer style
        
        Args:
            layer_name: Name of the layer this style is attached to
            
        Returns:
            ElementTree Element with the XML representation
        """
        # Create the root element
        root = ET.Element("asl")
        
        # Create the null descriptor (style identifier)
        null_node = ET.SubElement(root, "node")
        null_node.set("type", "Descriptor")
        null_node.set("classId", "null")
        null_node.set("name", "")
        
        # Add name with the (embedded) marker
        name_node = ET.SubElement(null_node, "node")
        name_node.set("key", "Nm  ")
        name_node.set("type", "Text")
        name_node.set("value", f"<{layer_name}> (embedded)")
        
        # Add UUID
        id_node = ET.SubElement(null_node, "node")
        id_node.set("key", "Idnt")
        id_node.set("type", "Text")
        id_node.set("value", self.uuid)
        
        # Create the style descriptor
        style_node = ET.SubElement(root, "node")
        style_node.set("type", "Descriptor")
        style_node.set("classId", "Styl")
        style_node.set("name", "")
        
        # Add document mode descriptor
        doc_mode = ET.SubElement(style_node, "node")
        doc_mode.set("key", "documentMode")
        doc_mode.set("type", "Descriptor")
        doc_mode.set("classId", "documentMode")
        doc_mode.set("name", "")
        
        # Add layer effects descriptor
        lefx_node = ET.SubElement(style_node, "node")
        lefx_node.set("key", "Lefx")
        lefx_node.set("type", "Descriptor")
        lefx_node.set("classId", "Lefx")
        lefx_node.set("name", "")
        
        # Add scale for effects
        scale_node = ET.SubElement(lefx_node, "node")
        scale_node.set("key", "Scl ")
        scale_node.set("type", "UnitFloat")
        scale_node.set("unit", "#Prc")
        scale_node.set("value", "100.0")
        
        # Add master switch
        master_switch = ET.SubElement(lefx_node, "node")
        master_switch.set("key", "masterFXSwitch")
        master_switch.set("type", "Boolean")
        master_switch.set("value", "1" if self.enabled else "0")
        
        # Add stroke effect if enabled
        if self.stroke_enabled:
            frfx_node = ET.SubElement(lefx_node, "node")
            frfx_node.set("key", "FrFX")
            frfx_node.set("type", "Descriptor")
            frfx_node.set("classId", "FrFX")
            frfx_node.set("name", "")
            
            # Enable the effect
            enable_node = ET.SubElement(frfx_node, "node")
            enable_node.set("key", "enab")
            enable_node.set("type", "Boolean")
            enable_node.set("value", "1")
            
            # Set style position (outside/inside/center)
            style_type_node = ET.SubElement(frfx_node, "node")
            style_type_node.set("key", "Styl")
            style_type_node.set("type", "Enum")
            style_type_node.set("typeId", "FStl")
            style_type_node.set("value", self.stroke_position)
            
            # Set fill type to solid color
            fill_type_node = ET.SubElement(frfx_node, "node")
            fill_type_node.set("key", "PntT")
            fill_type_node.set("type", "Enum")
            fill_type_node.set("typeId", "FrFl")
            fill_type_node.set("value", "SClr")  # Solid color
            
            # Set blend mode
            blend_mode_node = ET.SubElement(frfx_node, "node")
            blend_mode_node.set("key", "Md  ")
            blend_mode_node.set("type", "Enum")
            blend_mode_node.set("typeId", "BlnM")
            blend_mode_node.set("value", self.stroke_blend_mode)
            
            # Set opacity
            opacity_node = ET.SubElement(frfx_node, "node")
            opacity_node.set("key", "Opct")
            opacity_node.set("type", "UnitFloat")
            opacity_node.set("unit", "#Prc")
            opacity_node.set("value", str(self.stroke_opacity))
            
            # Set size
            size_node = ET.SubElement(frfx_node, "node")
            size_node.set("key", "Sz  ")
            size_node.set("type", "UnitFloat")
            size_node.set("unit", "#Pxl")
            size_node.set("value", str(self.stroke_size))
            
            # Set color
            color_node = ET.SubElement(frfx_node, "node")
            color_node.set("key", "Clr ")
            color_node.set("type", "Descriptor")
            color_node.set("classId", "RGBC")
            color_node.set("name", "")
            
            # Convert color to RGB values
            if isinstance(self.stroke_color, str):
                # Handle hex color
                color_str = self.stroke_color.lstrip('#')
                if len(color_str) == 6:
                    r = int(color_str[0:2], 16)
                    g = int(color_str[2:4], 16)
                    b = int(color_str[4:6], 16)
                else:
                    # Default to white if invalid hex
                    r, g, b = 255, 255, 255
            else:
                # Handle RGB tuple
                r, g, b = self.stroke_color
            
            # Red component
            red_node = ET.SubElement(color_node, "node")
            red_node.set("key", "Rd  ")
            red_node.set("type", "Double")
            red_node.set("value", str(float(r)))
            
            # Green component
            green_node = ET.SubElement(color_node, "node")
            green_node.set("key", "Grn ")
            green_node.set("type", "Double")
            green_node.set("value", str(float(g)))
            
            # Blue component
            blue_node = ET.SubElement(color_node, "node")
            blue_node.set("key", "Bl  ")
            blue_node.set("type", "Double")
            blue_node.set("value", str(float(b)))
        
        return root
    
    def to_asl_file(self, layer_name: str, output_path: str) -> None:
        """
        Save the layer style to an ASL file
        
        Args:
            layer_name: Name of the layer this style is attached to
            output_path: Path to save the ASL file
        """
        # Generate XML
        xml_root = self.generate_xml(layer_name)
        
        # Create temporary XML file
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as xml_file:
            xml_path = xml_file.name
            xml_str = ET.tostring(xml_root, encoding='unicode')
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
            xml_file.write(pretty_xml.encode('utf-8'))
        
        try:
            # Convert XML to ASL
            writer = ASLWriter(ByteOrder.BIG_ENDIAN)
            writer.write_file(xml_path, output_path)
                
        finally:
            # Clean up temporary files
            if os.path.exists(xml_path):
                os.remove(xml_path)
    
    def save_asl(self, layer_name: str, output_path: str) -> None:
        """
        Save the layer style to an ASL file
        
        Args:
            layer_name: Name of the layer this style is attached to
            output_path: Path to save the ASL file
        """
        from xml.dom import minidom
        
        root = self.to_xml(layer_name)
        
        # Convert to pretty XML
        xml_str = ET.tostring(root, encoding='unicode')
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)


@dataclass
class ShapeStyle:
    """Represents styling options for shapes"""
    fill: str = "none"  # Color or "none"
    stroke: str = "#000000"
    stroke_width: float = 1.0
    stroke_opacity: float = 1.0
    fill_opacity: float = 1.0
    stroke_linecap: str = "butt"  # butt, round, square
    stroke_linejoin: str = "miter"  # miter, round, bevel
    stroke_dasharray: Optional[str] = None  # e.g. "5,5" for dashed lines

@dataclass
class Shape:
    """Base class for all shapes"""
    style: ShapeStyle = field(default_factory=ShapeStyle)
    transform: str = ""  # SVG transform attribute
    
    def get_svg_attributes(self) -> Dict[str, str]:
        """Get common SVG attributes for the shape"""
        attrs = {
            'fill': self.style.fill,
            'stroke': self.style.stroke,
            'stroke-width': str(self.style.stroke_width),
            'stroke-opacity': str(self.style.stroke_opacity),
            'fill-opacity': str(self.style.fill_opacity),
            'stroke-linecap': self.style.stroke_linecap,
            'stroke-linejoin': self.style.stroke_linejoin
        }
        if self.style.stroke_dasharray:
            attrs['stroke-dasharray'] = self.style.stroke_dasharray
        if self.transform:
            attrs['transform'] = self.transform
        return attrs

    def to_svg_element(self) -> ET.Element:
        """Convert shape to SVG element - to be implemented by subclasses"""
        raise NotImplementedError
    
@dataclass
class Rectangle(Shape):
    """Rectangle shape"""
    x: float = 0
    y: float = 0
    width: float = 100
    height: float = 100
    rx: Optional[float] = None  # corner radius
    ry: Optional[float] = None  # corner radius
    
    def to_svg_element(self) -> ET.Element:
        attrs = self.get_svg_attributes()
        attrs.update({
            'x': str(self.x),
            'y': str(self.y),
            'width': str(self.width),
            'height': str(self.height)
        })
        if self.rx is not None:
            attrs['rx'] = str(self.rx)
        if self.ry is not None:
            attrs['ry'] = str(self.ry)
        return ET.Element('rect', attrs)

@dataclass
class Circle(Shape):
    """Circle shape"""
    cx: float = 0
    cy: float = 0
    r: float = 50
    
    def to_svg_element(self) -> ET.Element:
        attrs = self.get_svg_attributes()
        attrs.update({
            'cx': str(self.cx),
            'cy': str(self.cy),
            'r': str(self.r)
        })
        return ET.Element('circle', attrs)

@dataclass
class Ellipse(Shape):
    """Ellipse shape"""
    cx: float = 0
    cy: float = 0
    rx: float = 50
    ry: float = 30
    
    def to_svg_element(self) -> ET.Element:
        attrs = self.get_svg_attributes()
        attrs.update({
            'cx': str(self.cx),
            'cy': str(self.cy),
            'rx': str(self.rx),
            'ry': str(self.ry)
        })
        return ET.Element('ellipse', attrs)

@dataclass
class Line(Shape):
    """Line shape"""
    x1: float = 0
    y1: float = 0
    x2: float = 100
    y2: float = 100
    
    def to_svg_element(self) -> ET.Element:
        attrs = self.get_svg_attributes()
        attrs.update({
            'x1': str(self.x1),
            'y1': str(self.y1),
            'x2': str(self.x2),
            'y2': str(self.y2)
        })
        return ET.Element('line', attrs)

@dataclass
class Path(Shape):
    """Path shape"""
    d: str = ""  # SVG path data
    
    def to_svg_element(self) -> ET.Element:
        attrs = self.get_svg_attributes()
        attrs['d'] = self.d
        return ET.Element('path', attrs)

@dataclass
class ShapeGroup:
    """Group of shapes"""
    shapes: List[Shape]
    transform: str = ""
    
    def to_svg_element(self) -> ET.Element:
        group = ET.Element('g')
        if self.transform:
            group.set('transform', self.transform)
        for shape in self.shapes:
            group.append(shape.to_svg_element())
        return group
    

@dataclass
class TextStyle:
    """Represents text styling options for ShapeLayer"""
    font_family: str = "Segoe UI"
    font_size: int = 12
    fill_color: str = "#000000"
    stroke_color: str = "#000000"
    stroke_width: int = 0
    stroke_opacity: float = 0
    letter_spacing: int = 0
    word_spacing: int = 0
    text_align: str = "start"  # start, end, center, justify
    text_align_last: str = "auto"
    line_height: float = 1.2  # multiplier of font size
    use_rich_text: bool = False
    text_rendering: str = "auto"  # auto, optimizeSpeed, optimizeLegibility, geometricPrecision
    dominant_baseline: str = "middle"
    text_anchor: str = "middle" 
    paint_order: str = "stroke"
    stroke_linecap: str = "square"
    stroke_linejoin: str = "bevel"

@dataclass
class TextSpan:
    """Represents a span of text with position information"""
    text: str
    x: float = 0
    dy: Optional[float] = None  # vertical offset from previous line

@dataclass
class ShapeLayer:
    """Represents a text or vector shape layer with enhanced text styling"""
    content: Union[List[TextSpan], List[Union[Shape, ShapeGroup]], None] = None
    content_type: str = "text"
    name: str = "Vector Layer"
    visible: bool = True
    opacity: int = 255
    x: float = 0
    y: float = 0
    style: Union[TextStyle, ShapeStyle] = field(default_factory=TextStyle)
    layer_style: Optional[LayerStyle] = None

    @classmethod
    def from_text(cls, text: str, **kwargs) -> 'ShapeLayer':
        """Create a ShapeLayer from plain text, automatically splitting into spans"""
        lines = text.split('\n')
        spans = []
        for i, line in enumerate(lines):
            spans.append(TextSpan(
                text=line,
                x=0,
                dy=kwargs.get('style', TextStyle()).font_size * kwargs.get('style', TextStyle()).line_height if i > 0 else None
            ))
        return cls(content=spans, **kwargs)
    
    @classmethod
    def from_shapes(cls, shapes: Union[Shape, List[Shape], ShapeGroup], **kwargs) -> 'ShapeLayer':
        """Create a ShapeLayer from shapes"""
        if not isinstance(shapes, list):
            shapes = [shapes]
        return cls(content=shapes, content_type="shape", **kwargs)
    
    def _generate_svg_tspans(self) -> str:
        """Generate SVG tspan elements for each text span"""
        tspans = []
        for span in self.content:
            attributes = ['x="{}"'.format(span.x)]
            if span.dy is not None:
                attributes.append('dy="{}"'.format(span.dy))
            
            tspans.append(f'<tspan {" ".join(attributes)}>{span.text}</tspan>')
        return ''.join(tspans)
    
    def _generate_svg_shapes(self) -> ET.Element:
        """Generate SVG elements for shape content"""
        if self.content_type != "shape":
            raise ValueError("Cannot generate shapes for non-shape content")
        
        root = ET.Element('g')
        for shape in self.content:
            root.append(shape.to_svg_element())
        return root

    def get_svg_attributes(self) -> dict:
        """Get all SVG attributes based on content type"""
        if self.content_type == "text":
            return {
                'id': 'shape0',
                'krita:useRichText': str(self.style.use_rich_text).lower(),
                'text-rendering': self.style.text_rendering,
                'krita:textVersion': '3',
                'transform': f'translate({self.x}, {self.y})',
                'fill': self.style.fill_color,
                'stroke-opacity': str(self.style.stroke_opacity),
                'stroke': self.style.stroke_color,
                'stroke-width': str(self.style.stroke_width),
                'stroke-linecap': self.style.stroke_linecap,
                'stroke-linejoin': self.style.stroke_linejoin,
                'letter-spacing': str(self.style.letter_spacing),
                'word-spacing': str(self.style.word_spacing),
                'style': (
                    f'text-align: {self.style.text_align};'
                    f'text-align-last: {self.style.text_align_last};'
                    f'font-family: {self.style.font_family};'
                    f'font-size: {self.style.font_size};'
                    f'dominant-baseline: {self.style.dominant_baseline};'
                    f'text-anchor: {self.style.text_anchor};'
                    f'paint-order: {self.style.paint_order};'
                )
            }
        else:
            # For shape layers, we return minimal attributes
            return {
                'id': 'shape0',
                'transform': f'translate({self.x}, {self.y})'
            }

@dataclass
class PaintLayer:
    """Represents an image layer"""
    image: Union[str, Image.Image]  # Can be path or PIL Image
    name: Optional[str] = None
    visible: bool = True
    opacity: int = 255
    x: int = 0
    y: int = 0

class KritaDocument:
    """Main class for creating Krita documents"""
    
    def __init__(self, width: int = 1024, height: int = 1024):
        self.width = width
        self.height = height
        self.layers: List[Union[ShapeLayer, PaintLayer]] = []
        self.temp_dir = "krita_temp"
        self.layer_styles = {}  # Map layer UUIDs to layer styles
    
    def add_text_layer(self, text: Union[str, ShapeLayer], **kwargs) -> None:
        """Add a text layer to the document"""
        if isinstance(text, str):
            # Create a ShapeLayer from text string
            layer = ShapeLayer.from_text(text=text, **kwargs)
        elif isinstance(text, ShapeLayer):
            layer = text
        else:
            raise TypeError("text must be either a string or ShapeLayer object")
        
        self.layers.append(layer)

    def add_shape_layer(self, shapes: Union[Shape, List[Shape], ShapeGroup], **kwargs) -> None:
        """Add a shape layer to the document"""
        layer = ShapeLayer.from_shapes(shapes, **kwargs)
        self.layers.append(layer)
    
    def add_image_layer(self, image: Union[str, Image.Image], **kwargs) -> None:
        """Add an image layer to the document"""
        layer = PaintLayer(image=image, **kwargs)
        self.layers.append(layer)

    def _create_layer_styles_asl(self, zf) -> None:
        """Create and add layerstyles.asl to the zip file"""
        if not self.layer_styles:
            return
            
        asl_root = ET.Element("asl")
        
        # Add all layer styles to the XML
        for layer_uuid, (layer_style, layer_name) in self.layer_styles.items():
            # Use the actual layer name instead of a generic "Layer_uuid"
            style_xml = layer_style.generate_xml(layer_name)
            
            # Get the child nodes (null descriptor and style descriptor)
            for child in style_xml:
                asl_root.append(child)
        
        # Convert to pretty XML
        xml_str = ET.tostring(asl_root, encoding='unicode')
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

        # Save to temporary XML file
        xml_path = os.path.join(self.temp_dir, 'layerstyles.xml')
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        # Save to file in temp directory
        asl_path = os.path.join(self.temp_dir, 'annotations/layerstyles.asl')
        writer = ASLWriter(ByteOrder.BIG_ENDIAN)
        writer.write_file(xml_path, asl_path)
    

    def save(self, output_path: str) -> None:
        """Save the document as a .kra file"""
        try:
            # Create temporary directory structure
            os.makedirs(f"{self.temp_dir}/layers", exist_ok=True)
            os.makedirs(f"{self.temp_dir}/annotations", exist_ok=True)
            os.makedirs(f"{self.temp_dir}/animation", exist_ok=True)
            
            # Generate layer UUIDs and prepare layer info
            layer_info = []
            for i, layer in enumerate(self.layers, start=2):
                layer_uuid = "{" + str(uuid.uuid4()) + "}"
                layer_info.append((layer, layer_uuid, f"layer{i}"))
            
            with ZipFile(output_path, 'w', ZIP_STORED) as zf:
                # 1. Add mimetype (must be first)
                zf.writestr('mimetype', 'application/x-krita')
                
                # 2. Create and add documentinfo.xml
                doc_info = self._create_document_info()
                doc_info_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(doc_info, encoding='unicode')
                zf.writestr('documentinfo.xml', doc_info_str)
                
                # 3. Create and add maindoc.xml
                main_doc = self._create_main_doc(layer_info)
                main_doc_str = '<?xml version="1.0" encoding="UTF-8"?>\n'
                main_doc_str += '<!DOCTYPE DOC PUBLIC \'-//KDE//DTD krita 2.0//EN\' \'http://www.calligra.org/DTD/krita-2.0.dtd\'>\n'
                main_doc_str += ET.tostring(main_doc, encoding='unicode')
                zf.writestr('maindoc.xml', main_doc_str)
                
                # 4. Process each layer
                self._process_layers(zf, layer_info)

                # 5. Add layerstyles.asl
                if self.layer_styles:
                    self._create_layer_styles_asl(zf)
                
                # 6. Add animation metadata
                animation_meta = self._create_animation_metadata()
                zf.writestr(f"{self.temp_dir}/animation/index.xml", animation_meta)
                
                # 7. Add ICC profile
                with open('layer3.icc', 'rb') as f:  # Assuming ICC profile is in current directory
                    icc_data = f.read()
                    zf.writestr(f"{self.temp_dir}/annotations/icc", icc_data)
                
                # 8. Create and add preview
                self._create_preview(zf)
                
                # Add all files from temp directory to zip
                for root, dirs, files in os.walk(self.temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = file_path
                        zf.write(file_path, arcname)
            
            print(f"Successfully created Krita file: {output_path}")
            
        finally:
            # Clean up temporary files
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

    def _create_document_info(self) -> ET.Element:
        """Create the documentinfo.xml structure"""
        doc_info = ET.Element('document-info', {'xmlns': 'http://www.calligra.org/DTD/document-info'})
        about = ET.SubElement(doc_info, 'about')
        
        ET.SubElement(about, 'title')
        ET.SubElement(about, 'description')
        ET.SubElement(about, 'subject')
        abstract = ET.SubElement(about, 'abstract')
        abstract.text = '\n'
        ET.SubElement(about, 'keyword')
        creator = ET.SubElement(about, 'initial-creator')
        creator.text = 'Unknown'
        cycles = ET.SubElement(about, 'editing-cycles')
        cycles.text = '1'
        ET.SubElement(about, 'editing-time')
        date = ET.SubElement(about, 'date')
        date.text = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        creation = ET.SubElement(about, 'creation-date')
        creation.text = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        ET.SubElement(about, 'language')
        ET.SubElement(about, 'license')
        
        author = ET.SubElement(doc_info, 'author')
        ET.SubElement(author, 'full-name')
        ET.SubElement(author, 'creator-first-name')
        ET.SubElement(author, 'creator-last-name')
        ET.SubElement(author, 'initial')
        ET.SubElement(author, 'author-title')
        ET.SubElement(author, 'position')
        ET.SubElement(author, 'company')
        
        return doc_info

    def _create_main_doc(self, layer_info: List[Tuple]) -> ET.Element:
        """Create the maindoc.xml structure"""
        doc = ET.Element('DOC', {
            'xmlns': 'http://www.calligra.org/DTD/krita',
            'kritaVersion': '5.2.9',
            'syntaxVersion': '2.0',
            'editor': 'Krita'
        })
        
        image = ET.SubElement(doc, 'IMAGE', {
            'width': str(self.width),
            'mime': 'application/x-kra',
            'height': str(self.height),
            'description': '',
            'name': 'Unnamed',
            'y-res': '300',
            'colorspacename': 'RGBA',
            'x-res': '300',
            'profile': 'sRGB-elle-V2-srgbtrc.icc'
        })
        
        layers = ET.SubElement(image, 'layers')
        
        for layer, layer_uuid, layer_name in layer_info:
            attrs =  {
                'collapsed': '0',
                'channelflags': '',
                'visible': '1' if layer.visible else '0',
                'locked': '0',
                'y': str(layer.y),
                'filename': layer_name,
                'name': layer.name,
                'colorlabel': '0',
                'compositeop': 'normal',
                'x': str(layer.x),
                'uuid': layer_uuid,
                'intimeline': '0',
                'opacity': str(layer.opacity)
            }

            if hasattr(layer, 'layer_style') and layer.layer_style:
                attrs['layerstyle'] = "{" + layer.layer_style.uuid + "}"
                self.layer_styles[layer_uuid] = (layer.layer_style, layer.name)

            if isinstance(layer, ShapeLayer):
                attrs['nodetype'] = 'shapelayer'
                ET.SubElement(layers, 'layer', attrs)
            else:  # PaintLayer
                attrs['nodetype'] = 'paintlayer'
                attrs['colorspacename'] = 'RGBA'
                attrs['channellockflags'] = ''
                attrs['onionskin'] = '0'
                ET.SubElement(layers, 'layer', attrs)
        
        # Add additional required elements
        proj_bg = ET.SubElement(image, 'ProjectionBackgroundColor')
        proj_bg.set('ColorData', 'AAAAAA==')
        
        global_color = ET.SubElement(image, 'GlobalAssistantsColor')
        global_color.set('SimpleColorData', '176,176,176,255')
        
        mirror_axis = ET.SubElement(image, 'MirrorAxis')
        for elem_name, value in [
            ('mirrorHorizontal', '0'),
            ('mirrorVertical', '0'),
            ('lockHorizontal', '0'),
            ('lockVertical', '0'),
            ('hideHorizontalDecoration', '0'),
            ('hideVerticalDecoration', '0'),
            ('handleSize', '32'),
            ('horizontalHandlePosition', '64'),
            ('verticalHandlePosition', '64')
        ]:
            elem = ET.SubElement(mirror_axis, elem_name)
            elem.set('value', value)
            elem.set('type', 'value')
        
        axis_pos = ET.SubElement(mirror_axis, 'axisPosition')
        axis_pos.set('y', str(self.height//2))
        axis_pos.set('x', str(self.width//2))
        axis_pos.set('type', 'pointf')
        
        ET.SubElement(image, 'Palettes')
        ET.SubElement(image, 'resources')
        
        # Add animation section
        animation = ET.SubElement(image, 'animation')
        framerate = ET.SubElement(animation, 'framerate')
        framerate.set('value', '24')
        framerate.set('type', 'value')
        
        range_elem = ET.SubElement(animation, 'range')
        range_elem.set('from', '0')
        range_elem.set('to', '100')
        range_elem.set('type', 'timerange')
        
        current_time = ET.SubElement(animation, 'currentTime')
        current_time.set('value', '0')
        current_time.set('type', 'value')
        
        return doc

    def _create_svg_content(self, layer: ShapeLayer) -> str:
        """Create SVG content for shape layer"""
        # Define all namespaces
        namespaces = {
            'xmlns': 'http://www.w3.org/2000/svg',
            'xmlns:xlink': 'http://www.w3.org/1999/xlink',
            'xmlns:krita': 'http://krita.org/namespaces/svg/krita',
            'xmlns:sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'
        }
        
        # Create root SVG element with namespaces
        svg = ET.Element('svg', namespaces)
        svg.set('width', f"{self.width}")
        svg.set('height', f"{self.height}")
        svg.set('viewBox', f"0 0 {self.width} {self.height}")
        
        if layer.content_type == "text":
            # Create text element directly using ElementTree
            attrs = layer.get_svg_attributes()
            text_elem = ET.SubElement(svg, 'text')
            
            # Set all attributes
            for k, v in attrs.items():
                if ':' in k:  # Handle namespaced attributes
                    ET.register_namespace('krita', 'http://krita.org/namespaces/svg/krita')
                text_elem.set(k, v)
                
            # Add tspans
            for span in layer.content:
                tspan = ET.SubElement(text_elem, 'tspan')
                if span.x is not None:
                    tspan.set('x', str(span.x))
                if span.dy is not None:
                    tspan.set('dy', str(span.dy))
                tspan.text = span.text
        else:
            group = ET.SubElement(svg, 'g', layer.get_svg_attributes())
            shapes_group = layer._generate_svg_shapes()
            for child in shapes_group:
                group.append(child)
        
        # Generate the complete SVG document
        svg_str = f'''<?xml version="1.0" standalone="no"?>
    <!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN" "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">
    <!-- Created using Krita: https://krita.org -->
    {ET.tostring(svg, encoding='unicode')}'''
        
        return svg_str

    def _create_animation_metadata(self) -> str:
        """Create the animation metadata XML"""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<animation-metadata xmlns="http://www.calligra.org/DTD/krita">
<framerate type="value" value="24"/>
<range from="0" type="timerange" to="100"/>
<currentTime type="value" value="0"/>
<export-settings>
<sequenceFilePath type="value" value=""/>
<sequenceBaseName type="value" value=""/>
<sequenceInitialFrameNumber type="value" value="-1"/>
</export-settings>
</animation-metadata>'''

    def _create_preview(self, zf: ZipFile) -> None:
        """Create and add preview.png to the zip file"""
        if len(self.layers) > 0 and isinstance(self.layers[-1], PaintLayer):
            # Use the last paint layer as preview
            if isinstance(self.layers[-1].image, str):
                preview = Image.open(self.layers[-1].image)
            else:
                preview = self.layers[-1].image.copy()
        else:
            # Create blank preview
            preview = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        
        # Resize to thumbnail size
        preview.thumbnail((256, 256))
        preview_path = os.path.join(self.temp_dir, 'preview.png')
        preview.save(preview_path, 'PNG')
        zf.write(preview_path, 'preview.png')

        # delete the preview file
        os.remove(preview_path)

    def _process_layers(self, zf: ZipFile, layer_info: List[Tuple]) -> None:
        """Process and add all layers to the zip file"""
        for layer, _, layer_name in layer_info:
            if isinstance(layer, ShapeLayer):
                self._add_shape_layer(zf, layer, layer_name)
            elif isinstance(layer, PaintLayer):
                self._add_paint_layer(zf, layer, layer_name)

    def _add_shape_layer(self, zf: ZipFile, layer: ShapeLayer, layer_name: str) -> None:
        """Add a shape layer to the zip file"""
        # Create shape layer directory in temp dir
        shape_dir = os.path.join(self.temp_dir, 'layers', f'{layer_name}.shapelayer')
        os.makedirs(shape_dir, exist_ok=True)
        
        # Create and save SVG content
        svg_content = self._create_svg_content(layer)
        svg_path = os.path.join(shape_dir, 'content.svg')
        
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
            
        # Add to zip
        # zf.write(svg_path, os.path.join('layers', f'{layer_name}.shapelayer', 'content.svg'))

        
    def _add_paint_layer(self, zf: ZipFile, layer: PaintLayer, layer_name: str) -> None:
        """Add a paint layer to the zip file"""
        # Load image if path provided
        if isinstance(layer.image, str):
            img = Image.open(layer.image)
        else:
            img = layer.image
            
        # Convert to RGBA
        img = img.convert('RGBA')
        
        # Save layer data
        layer_path = os.path.join(self.temp_dir, 'layers', layer_name)
        self._save_krita_layer(img, layer_path)
        
        # Add default pixel data
        defaultpixel_path = f"{layer_path}.defaultpixel"
        with open(defaultpixel_path, 'wb') as f:
            f.write(bytes([0, 0, 0, 0]))
            
        # Add ICC profile
        icc_path = f"{layer_path}.icc"
        with open('layer3.icc', 'rb') as src, open(icc_path, 'wb') as dst:
            dst.write(src.read())
            
        # Add to zip
        # zf.write(layer_path, os.path.join('layers', layer_name))
        # zf.write(defaultpixel_path, os.path.join('layers', f'{layer_name}.defaultpixel'))
        # zf.write(icc_path, os.path.join('layers', f'{layer_name}.icc'))

    @staticmethod
    def _save_krita_layer(img: Image.Image, output_path: str) -> None:
        """Save an image as a Krita layer file"""
        w, h = img.size
        nx = math.ceil(w / 64)
        ny = math.ceil(h / 64)
        
        tile_entries = []
        for ty in range(ny):
            for tx in range(nx):
                left = tx * 64
                top = ty * 64
                
                # Create tile
                tile_img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
                crop = img.crop((left, top, left + 64, top + 64))
                tile_img.paste(crop, (0, 0))
                
                # Convert to numpy array
                arr = np.array(tile_img, dtype=np.uint8)
                
                # Create planar data (BGRA order)
                planes = [
                    arr[:, :, 2].flatten(),  # Blue
                    arr[:, :, 1].flatten(),  # Green
                    arr[:, :, 0].flatten(),  # Red
                    arr[:, :, 3].flatten(),  # Alpha
                ]
                plane_data = np.concatenate(planes).tobytes()
                
                # Compress tile
                from lzf import compress
                compressed = compress(plane_data)
                if compressed is None:
                    compressed = b""
                
                # Create tile data
                tile_data = b"\x01" + compressed
                tile_header = f"{left},{top},LZF,{len(tile_data)}\n".encode("utf-8")
                tile_entries.append((tile_header, tile_data))
        
        # Write layer file
        header = [
            b"VERSION 2\n",
            b"TILEWIDTH 64\n",
            b"TILEHEIGHT 64\n",
            b"PIXELSIZE 4\n",
            f"DATA {len(tile_entries)}\n".encode("utf-8"),
        ]
        
        with open(output_path, "wb") as f:
            f.write(b"".join(header))
            for header, data in tile_entries:
                f.write(header)
                f.write(data)