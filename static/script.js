const uploadForm = document.getElementById('uploadForm');
const imageInput = document.getElementById('imageInput');
const submitBtn = document.getElementById('submitBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessage = document.getElementById('errorMessage');
const resultsSection = document.getElementById('resultsSection');
const resultsBody = document.getElementById('resultsBody');
const toggleColumns = document.getElementById('toggleColumns');

//API endpoint
const API_URL = 'http://localhost:8000';

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const file = imageInput.files[0];
    if (!file) {
        showError('Please select an image file');
        return;
    }
    
    hideError();
    hideResults();
    showLoading();
    submitBtn.disabled = true;
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to analyze image');
        }
        
        const data = await response.json();
        
        if (data.success && data.results) {
            displayResults(data.results);
        } else {
            throw new Error('Invalid response from server');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred while processing the image. Please try again.');
    } finally {
        hideLoading();
        submitBtn.disabled = false;
    }
});

function showLoading() {
    loadingIndicator.classList.remove('hidden');
}

function hideLoading() {
    loadingIndicator.classList.add('hidden');
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}

function showResults() {
    resultsSection.classList.remove('hidden');
}

function hideResults() {
    resultsSection.classList.add('hidden');
}

function displayResults(results) {
    resultsBody.innerHTML = '';
    
    const sortedResults = results.sort((a, b) => {
        const order = [
            "Leukocytes", "Nitrite", "Urobilinogen", "Protein", "pH",
            "Blood", "Specific Gravity", "Ketone", "Bilirubin", "Calcium"
        ];
        return order.indexOf(a.parameter) - order.indexOf(b.parameter);
    });
    
    sortedResults.forEach(result => {
        const row = document.createElement('tr');
        
        const paramCell = document.createElement('td');
        paramCell.className = 'parameter-name';
        paramCell.textContent = result.parameter;
        
        const valueCell = document.createElement('td');
        valueCell.textContent = result.value || 'N/A';
        
        const normalRangeCell = document.createElement('td');
        normalRangeCell.textContent = result.normal_range || 'N/A';
        
        const statusCell = document.createElement('td');
        const statusBadge = document.createElement('span');
        const statusValue = result.status || 'Unknown';
        statusBadge.className = `status-badge status-${statusValue.toLowerCase()}`;
        statusBadge.textContent = statusValue;
        statusCell.appendChild(statusBadge);
        
        const confidenceCell = document.createElement('td');
        confidenceCell.className = 'technical-column confidence-column';
        
        if (result.yolo_confidence !== null && result.yolo_confidence !== undefined) {
            const percentage = Math.round(result.yolo_confidence * 100);
            confidenceCell.textContent = `${percentage}%`;
        } else {
            confidenceCell.textContent = 'N/A';
        }
        
        const deltaECell = document.createElement('td');
        deltaECell.className = 'technical-column';
        
        if (result.delta_e !== undefined) {
            const deltaEBadge = document.createElement('span');
            const colorConf = result.color_confidence || 'low';
            deltaEBadge.className = `confidence-badge confidence-${colorConf}`;
            deltaEBadge.textContent = result.delta_e;
            deltaECell.appendChild(deltaEBadge);
        } else if (result.reason) {
            deltaECell.textContent = result.reason;
        } else {
            deltaECell.textContent = 'N/A';
        }
        
        row.appendChild(paramCell);
        row.appendChild(valueCell);
        row.appendChild(normalRangeCell);
        row.appendChild(statusCell);
        row.appendChild(confidenceCell);
        row.appendChild(deltaECell);
        
        resultsBody.appendChild(row);
    });
    
    toggleTechnicalColumns();
    
    showResults();
}

function toggleTechnicalColumns() {
    const isChecked = toggleColumns.checked;
    const technicalColumns = document.querySelectorAll('.technical-column');
    technicalColumns.forEach(col => {
        if (isChecked) {
            col.classList.remove('hidden-column');
        } else {
            col.classList.add('hidden-column');
        }
    });
}

if (toggleColumns) {
    toggleColumns.addEventListener('change', toggleTechnicalColumns);
}

imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const labelText = document.querySelector('.file-label-text');
        labelText.textContent = file.name;
    }
});

