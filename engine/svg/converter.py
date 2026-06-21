"""SVG to PNG converter module."""
import os
import io
try:
    import cairosvg
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False


class SVGConverter:
    """Convert SVG files to PNG format."""
    
    @staticmethod
    def svg_to_png(svg_content: str, output_path: str, dpi: int = 96) -> bool:
        """Convert SVG string to PNG file.
        
        Args:
            svg_content: SVG content as string
            output_path: Path to save PNG file
            dpi: DPI for conversion (default 96)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not HAS_CAIROSVG:
            print("Warning: cairosvg not installed. Install with: pip install cairosvg")
            return False
        
        try:
            # Convert SVG string to PNG bytes
            png_bytes = cairosvg.svg2png(
                bytestring=svg_content.encode('utf-8'),
                dpi=dpi
            )
            
            # Write PNG to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(png_bytes)
            
            return True
        except Exception as e:
            print(f"Error converting SVG to PNG: {e}")
            return False
    
    @staticmethod
    def svg_file_to_png(svg_path: str, output_path: str, dpi: int = 96) -> bool:
        """Convert SVG file to PNG file.
        
        Args:
            svg_path: Path to SVG file
            output_path: Path to save PNG file
            dpi: DPI for conversion (default 96)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not HAS_CAIROSVG:
            print("Warning: cairosvg not installed. Install with: pip install cairosvg")
            return False
        
        try:
            # Read SVG file
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg_content = f.read()
            
            # Convert using svg_to_png
            return SVGConverter.svg_to_png(svg_content, output_path, dpi)
        except Exception as e:
            print(f"Error reading SVG file: {e}")
            return False
