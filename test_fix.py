import sys
import os
import requests
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

with open("test_result.txt", "w") as f:
    f.write("Starting test_fix.py execution...\n")
    try:
        from data.fakeddit_preprocessor_image import ImageProcessor
        f.write("Successfully imported ImageProcessor\n")
    except ImportError as e:
        f.write(f"Failed to import ImageProcessor: {e}\n")
        sys.exit(1)

    try:
        processor = ImageProcessor()
        
        # Test is_video_url
        f.write("\nTesting is_video_url...\n")
        if not hasattr(processor, 'is_video_url'):
            f.write("FAIL: is_video_url method missing\n")
        else:
            test_urls = [
                ("http://example.com/video.mp4", True),
                ("http://youtube.com/watch?v=123", True),
                ("http://example.com/image.jpg", False),
                ("", False)
            ]
            for url, expected in test_urls:
                result = processor.is_video_url(url)
                status = "PASS" if result == expected else "FAIL"
                f.write(f"  {status}: {url} -> {result}\n")

        # Test process_image existence
        f.write("\nTesting process_image...\n")
        if not hasattr(processor, 'process_image'):
            f.write("FAIL: process_image method missing\n")
        else:
            f.write("PASS: process_image method exists\n")

        # Test _download_image existence
        f.write("\nTesting _download_image...\n")
        if not hasattr(processor, '_download_image'):
            f.write("FAIL: _download_image method missing\n")
        else:
            f.write("PASS: _download_image method exists\n")
            
    except Exception as e:
        f.write(f"\nERROR running tests: {e}\n")
