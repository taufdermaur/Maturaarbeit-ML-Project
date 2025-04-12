#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def convert_eps_to_raster(eps_file, output_format="png", dpi=300):
    """
    Convert an EPS file to raster format using Pillow and Ghostscript
    
    Args:
        eps_file (str): Path to the EPS file
        output_format (str): Output format (png, jpg, tiff, etc.)
        dpi (int): Resolution in dots per inch
        
    Returns:
        str: Path to the converted file or None if conversion failed
    """
    try:
        # Create output path
        output_dir = Path(eps_file).parent / f"converted_{output_format}"
        output_dir.mkdir(exist_ok=True)
        
        # Set output filename
        output_file = output_dir / f"{Path(eps_file).stem}.{output_format}"
        
        # Open the EPS file with Pillow
        # Note: This requires Ghostscript to be installed
        img = Image.open(eps_file)
        
        # Calculate new size based on DPI
        if hasattr(img, 'info') and 'dpi' in img.info:
            orig_dpi = img.info['dpi']
            scale_factor = dpi / orig_dpi[0]
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save as the specified format
        img.save(str(output_file), dpi=(dpi, dpi))
        img.close()
        
        return str(output_file)
    
    except Exception as e:
        logger.error(f"Error converting {eps_file}: {str(e)}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert EPS files to raster formats')
    parser.add_argument('--format', type=str, default='png', 
                        help='Output format (png, jpg, tiff, etc.). Default: png')
    parser.add_argument('--dpi', type=int, default=300, 
                        help='Output resolution in DPI. Default: 300')
    parser.add_argument('--dir', type=str, default='.', 
                        help='Directory containing EPS files. Default: current directory')
    
    args = parser.parse_args()
    
    # Find all EPS files in the specified directory
    eps_files = list(Path(args.dir).glob('*.eps'))
    total_files = len(eps_files)
    
    if total_files == 0:
        logger.info(f"No EPS files found in directory: {args.dir}")
        return
    
    logger.info(f"Found {total_files} EPS files to convert to {args.format} format at {args.dpi} DPI")
    logger.info(f"Output will be saved to: {Path(args.dir) / f'converted_{args.format}'}")
    logger.info("-" * 60)
    
    # Process each file
    successful = 0
    
    for eps_file in eps_files:
        logger.info(f"Converting {eps_file.name} to {eps_file.stem}.{args.format}... ")
        result = convert_eps_to_raster(str(eps_file), args.format, args.dpi)
        
        if result:
            logger.info("Done")
            successful += 1
        else:
            logger.info("Failed")
    
    logger.info("-" * 60)
    logger.info(f"Conversion complete: {successful} of {total_files} files converted successfully")
    logger.info(f"Output saved to: {Path(args.dir) / f'converted_{args.format}'}")

if __name__ == "__main__":
    main()
