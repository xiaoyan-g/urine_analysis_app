import pytest
import cv2
import numpy as np
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from urine_analysis_processor2 import (
    preprocess_image,
    create_pad_mask,
    segment_pads,
    extract_lab_color,
    load_reference_chart,
    estimate_biomarker,
    save_debug_crops,
    analyze_urine_strip,
    PAD_ORDER,
    DELTA_E_THRESHOLD,
    CSV_PATH
)


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[20:80, 20:180] = [100, 150, 200]  # Add some colored region
    return img


@pytest.fixture
def sample_image_path(sample_image, tmp_path):
    """Save sample image to temporary file."""
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), sample_image)
    return str(img_path)


@pytest.fixture
def sample_roi():
    """Create a sample ROI with colored pad region."""
    roi = np.zeros((50, 50, 3), dtype=np.uint8)
    roi[10:40, 10:40] = [120, 140, 160]  # Colored pad region
    return roi


@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model."""
    model = Mock()
    
    # Create mock detection boxes
    mock_box1 = Mock()
    mock_box1.xyxy = np.array([[10, 10, 30, 30]], dtype=np.float32)
    mock_box1.cls = np.array([0], dtype=np.int64)  # Leukocytes
    mock_box1.conf = np.array([0.9], dtype=np.float32)
    
    mock_box2 = Mock()
    mock_box2.xyxy = np.array([[40, 10, 60, 30]], dtype=np.float32)
    mock_box2.cls = np.array([1], dtype=np.int64)  # Nitrite
    mock_box2.conf = np.array([0.85], dtype=np.float32)
    
    mock_detections = Mock()
    mock_detections.boxes = [mock_box1, mock_box2]
    
    mock_result = Mock()
    mock_result.boxes = mock_detections
    
    mock_results = [mock_result]
    
    model.predict.return_value = mock_results
    return model


@pytest.fixture
def sample_pad_data(sample_roi):
    """Create sample pad data structure."""
    mask = create_pad_mask(sample_roi)
    return {
        "parameter": "Leukocytes",
        "roi": sample_roi,
        "mask": mask,
        "yolo_confidence": 0.9
    }


class TestPreprocessImage:
    """Tests for preprocess_image function."""
    
    def test_preprocess_image_success(self, sample_image_path):
        """Test successful image loading."""
        img = preprocess_image(sample_image_path)
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3  # Should be BGR image
    
    def test_preprocess_image_file_not_found(self):
        """Test error handling for non-existent file (Ultralytics patches cv2.imread and raises FileNotFoundError)."""
        with pytest.raises((ValueError, FileNotFoundError)):
            preprocess_image("nonexistent_file.jpg")
    
    def test_preprocess_image_invalid_file(self, tmp_path):
        """Test error handling for invalid image file."""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("not an image")
        
        with pytest.raises(ValueError, match="Image not loaded"):
            preprocess_image(str(invalid_file))


class TestCreatePadMask:
    """Tests for create_pad_mask function."""
    
    def test_create_pad_mask_basic(self, sample_roi):
        """Test mask creation with valid ROI."""
        mask = create_pad_mask(sample_roi)
        
        assert mask is not None
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8
        assert mask.shape[:2] == sample_roi.shape[:2]
        assert mask.max() <= 255
        assert mask.min() >= 0
    
    def test_create_pad_mask_returns_binary(self, sample_roi):
        """Test that mask contains only 0 and 255 values."""
        mask = create_pad_mask(sample_roi)
        unique_values = np.unique(mask)
        assert all(val in [0, 255] for val in unique_values)
    
    def test_create_pad_mask_dark_image(self):
        """Test mask creation with very dark image."""
        dark_roi = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = create_pad_mask(dark_roi)
        # Should exclude very dark pixels
        assert mask is not None


class TestSegmentPads:
    """Tests for segment_pads function."""
    
    @patch('urine_analysis_processor2.MODEL')
    def test_segment_pads_success(self, mock_model, sample_image):
        """Test successful pad segmentation."""
        # Setup mock: segment_pads expects results[0].boxes to be a list
        mock_box1 = Mock()
        mock_box1.xyxy = np.array([[10, 10, 30, 30]], dtype=np.float32)
        mock_box1.cls = np.array([0], dtype=np.int64)
        mock_box1.conf = np.array([0.9], dtype=np.float32)
        
        mock_result = Mock()
        mock_result.boxes = [mock_box1]
        
        mock_model.predict.return_value = [mock_result]
        
        pads = segment_pads(sample_image)
        
        assert len(pads) > 0
        assert all("parameter" in pad for pad in pads)
        assert all("roi" in pad for pad in pads)
        assert all("mask" in pad for pad in pads)
        assert all("yolo_confidence" in pad for pad in pads)
    
    @patch('urine_analysis_processor2.MODEL')
    def test_segment_pads_no_detections(self, mock_model, sample_image):
        """Test error handling when no pads are detected."""
        mock_result = Mock()
        mock_result.boxes = []
        
        mock_model.predict.return_value = [mock_result]
        
        with pytest.raises(ValueError, match="No test pads detected"):
            segment_pads(sample_image)
    
    @patch('urine_analysis_processor2.MODEL')
    def test_segment_pads_sorted_by_x(self, mock_model, sample_image):
        """Test that pads are sorted by x-position."""
        # Create boxes with different x positions; results[0].boxes must be a list
        mock_box1 = Mock()
        mock_box1.xyxy = np.array([[50, 10, 70, 30]], dtype=np.float32)
        mock_box1.cls = np.array([1], dtype=np.int64)
        mock_box1.conf = np.array([0.9], dtype=np.float32)
        
        mock_box2 = Mock()
        mock_box2.xyxy = np.array([[10, 10, 30, 30]], dtype=np.float32)
        mock_box2.cls = np.array([0], dtype=np.int64)
        mock_box2.conf = np.array([0.9], dtype=np.float32)
        
        mock_result = Mock()
        mock_result.boxes = [mock_box1, mock_box2]
        
        mock_model.predict.return_value = [mock_result]
        
        pads = segment_pads(sample_image)
        
        # Check that pads are sorted (first pad should have smaller x1)
        if len(pads) >= 2:
            # Extract x coordinates from ROI positions (approximate check)
            assert pads[0]["parameter"] == "Leukocytes"  # Should be first after sorting


class TestExtractLabColor:
    """Tests for extract_lab_color function."""
    
    def test_extract_lab_color_success(self, sample_pad_data):
        """Test successful color extraction."""
        lab_color, status = extract_lab_color(sample_pad_data)
        
        assert status == "good"
        assert lab_color is not None
        assert isinstance(lab_color, np.ndarray)
        assert len(lab_color) == 3  # L, a, b values
    
    def test_extract_lab_color_low_pixels(self):
        """Test color extraction with insufficient pixels."""
        # Create ROI with very few valid pixels
        small_roi = np.zeros((5, 5, 3), dtype=np.uint8)
        small_mask = np.zeros((5, 5), dtype=np.uint8)
        small_mask[2, 2] = 255  # Only one pixel
        
        pad_data = {
            "roi": small_roi,
            "mask": small_mask
        }
        
        lab_color, status = extract_lab_color(pad_data, min_pixels=500)
        
        assert status == "low_pixels"
        assert lab_color is None
    
    def test_extract_lab_color_high_variance(self):
        """Test color extraction with high variance."""
        # Create ROI with high color variance
        high_var_roi = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        high_var_mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        pad_data = {
            "roi": high_var_roi,
            "mask": high_var_mask
        }
        
        lab_color, status = extract_lab_color(pad_data)
        
        # May return high_variance or good depending on actual variance
        assert status in ["high_variance", "good"]


class TestLoadReferenceChart:
    """Tests for load_reference_chart function."""
    
    def test_load_reference_chart_success(self):
        """Test successful loading of reference chart."""
        if not os.path.exists(CSV_PATH):
            pytest.skip(f"CSV file not found: {CSV_PATH}")
        
        refs = load_reference_chart(CSV_PATH)
        
        assert isinstance(refs, dict)
        assert len(refs) > 0
        
        # Check that all expected parameters are present
        for param in PAD_ORDER:
            assert param in refs
            assert len(refs[param]) > 0
    
    def test_load_reference_chart_structure(self):
        """Test that reference chart has correct structure."""
        if not os.path.exists(CSV_PATH):
            pytest.skip(f"CSV file not found: {CSV_PATH}")
        
        refs = load_reference_chart(CSV_PATH)
        
        for param, lab_list in refs.items():
            assert isinstance(lab_list, list)
            for entry in lab_list:
                assert len(entry) == 4  # (lab, value, normal_range, status)
                lab, value, normal_range, status = entry
                assert isinstance(lab, np.ndarray)
                assert len(lab) == 3  # L, a, b values


class TestEstimateBiomarker:
    """Tests for estimate_biomarker function."""
    
    def test_estimate_biomarker_success(self):
        """Test successful biomarker estimation."""
        # Use a known LAB color from reference chart
        if not os.path.exists(CSV_PATH):
            pytest.skip(f"CSV file not found: {CSV_PATH}")
        
        # Load reference chart to get a known color
        refs = load_reference_chart(CSV_PATH)
        if "Leukocytes" in refs and len(refs["Leukocytes"]) > 0:
            known_lab, known_value, _, _ = refs["Leukocytes"][0]
            
            value, dist, normal_range, status = estimate_biomarker(known_lab, "Leukocytes")
            
            assert value is not None
            assert isinstance(dist, (int, float))
            assert dist >= 0  # Distance should be non-negative
            assert normal_range is not None
            assert status is not None
    
    def test_estimate_biomarker_unknown_parameter(self):
        """Test biomarker estimation with unknown parameter."""
        test_lab = np.array([50.0, 0.0, 0.0])
        value, dist, normal_range, status = estimate_biomarker(test_lab, "UnknownParameter")
        
        assert value == "unknown_param"
        assert dist == 0.0
    
    def test_estimate_biomarker_unclear_match(self):
        """Test biomarker estimation with high delta E."""
        # Use a color very different from reference
        far_lab = np.array([200.0, 200.0, 200.0])
        
        value, dist, normal_range, status = estimate_biomarker(far_lab, "Leukocytes")
        
        # Should return unclear if distance is too high
        if dist > DELTA_E_THRESHOLD:
            assert "(unclear)" in str(value)


class TestSaveDebugCrops:
    """Tests for save_debug_crops function."""
    
    def test_save_debug_crops_success(self, sample_pad_data, tmp_path):
        """Test successful saving of debug crops."""
        pads = [sample_pad_data]
        debug_folder = str(tmp_path / "debug_test")
        os.makedirs(debug_folder, exist_ok=True)
        
        result_folder = save_debug_crops(pads, debug_folder)
        
        assert result_folder == debug_folder
        assert os.path.exists(debug_folder)
        
        # Check that files were created
        files = os.listdir(debug_folder)
        assert len(files) > 0
    
    def test_save_debug_crops_multiple_pads(self, sample_roi, tmp_path):
        """Test saving crops for multiple pads."""
        pad1 = {
            "parameter": "Leukocytes",
            "roi": sample_roi,
            "mask": create_pad_mask(sample_roi),
            "yolo_confidence": 0.9
        }
        pad2 = {
            "parameter": "Nitrite",
            "roi": sample_roi,
            "mask": create_pad_mask(sample_roi),
            "yolo_confidence": 0.85
        }
        
        pads = [pad1, pad2]
        debug_folder = str(tmp_path / "debug_test")
        os.makedirs(debug_folder, exist_ok=True)
        
        save_debug_crops(pads, debug_folder)
        
        files = os.listdir(debug_folder)
        # Should have 3 files per pad (raw, mask, masked)
        assert len(files) >= 6


class TestAnalyzeUrineStrip:
    """Tests for analyze_urine_strip function."""
    
    @patch('urine_analysis_processor2.MODEL')
    @patch('urine_analysis_processor2.save_debug_crops')
    def test_analyze_urine_strip_success(self, mock_save_debug, mock_model, sample_image_path):
        """Test successful full pipeline analysis."""
        # Setup mock YOLO model: results[0].boxes must be a list
        mock_box = Mock()
        mock_box.xyxy = np.array([[10, 10, 30, 30]], dtype=np.float32)
        mock_box.cls = np.array([0], dtype=np.int64)
        mock_box.conf = np.array([0.9], dtype=np.float32)
        
        mock_result = Mock()
        mock_result.boxes = [mock_box]
        
        mock_model.predict.return_value = [mock_result]
        mock_save_debug.return_value = "debug_folder"
        
        output = analyze_urine_strip(sample_image_path, save_debug=True)
        
        assert isinstance(output, dict)
        assert "results" in output
        assert "annotated_image" in output
        results = output["results"]
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check result structure
        for result in results:
            assert "parameter" in result
            assert "value" in result
            assert "delta_e" in result or result.get("reason") is not None
            assert "yolo_confidence" in result
            assert "color_confidence" in result
            assert "normal_range" in result
            assert "status" in result
    
    @patch('urine_analysis_processor2.MODEL')
    def test_analyze_urine_strip_no_debug(self, mock_model, sample_image_path):
        """Test analysis without debug saving."""
        # Setup mock YOLO model: results[0].boxes must be a list
        mock_box = Mock()
        mock_box.xyxy = np.array([[10, 10, 30, 30]], dtype=np.float32)
        mock_box.cls = np.array([0], dtype=np.int64)
        mock_box.conf = np.array([0.9], dtype=np.float32)
        
        mock_result = Mock()
        mock_result.boxes = [mock_box]
        
        mock_model.predict.return_value = [mock_result]
        
        output = analyze_urine_strip(sample_image_path, save_debug=False)
        
        assert isinstance(output, dict)
        assert "results" in output
        assert "annotated_image" in output
        results = output["results"]
        assert isinstance(results, list)
    
    def test_analyze_urine_strip_invalid_image(self):
        """Test analysis with invalid image path (raises FileNotFoundError when Ultralytics patches cv2.imread)."""
        with pytest.raises((ValueError, FileNotFoundError)):
            analyze_urine_strip("nonexistent_image.jpg")


class TestConstants:
    """Tests for module constants."""
    
    def test_pad_order_defined(self):
        """Test that PAD_ORDER is properly defined."""
        assert isinstance(PAD_ORDER, list)
        assert len(PAD_ORDER) > 0
        assert all(isinstance(pad, str) for pad in PAD_ORDER)
    
    def test_delta_e_threshold_defined(self):
        """Test that DELTA_E_THRESHOLD is properly defined."""
        assert isinstance(DELTA_E_THRESHOLD, (int, float))
        assert DELTA_E_THRESHOLD > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
