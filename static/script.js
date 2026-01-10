const uploadForm = document.getElementById('uploadForm');
const imageInput = document.getElementById('imageInput');
const submitBtn = document.getElementById('submitBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessage = document.getElementById('errorMessage');
const resultsSection = document.getElementById('resultsSection');
const resultsBody = document.getElementById('resultsBody');

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
    // Clear previous results
    resultsBody.innerHTML = '';
    
    // Sort results by parameter order to maintain consistency
    const sortedResults = results.sort((a, b) => {
        const order = [
            "Leukocytes", "Nitrite", "Urobilinogen", "Protein", "pH",
            "Blood", "Specific Gravity", "Ketone", "Bilirubin", "Calcium"
        ];
        return order.indexOf(a.parameter) - order.indexOf(b.parameter);
    });
    
    sortedResults.forEach(result => {
        const row = document.createElement('tr');
        
        // Parameter name
        const paramCell = document.createElement('td');
        paramCell.className = 'parameter-name';
        paramCell.textContent = result.parameter;
        
        // Value
        const valueCell = document.createElement('td');
        valueCell.textContent = result.value || 'N/A';
        
        // Confidence badge
        const confidenceCell = document.createElement('td');
        const confidenceBadge = document.createElement('span');
        confidenceBadge.className = `confidence-badge confidence-${result.confidence}`;
        confidenceBadge.textContent = result.confidence || 'unknown';
        confidenceCell.appendChild(confidenceBadge);
        
        // Delta E
        const deltaECell = document.createElement('td');
        deltaECell.className = 'delta-e';
        if (result.delta_e !== undefined) {
            deltaECell.textContent = result.delta_e;
        } else if (result.reason) {
            deltaECell.textContent = result.reason;
        } else {
            deltaECell.textContent = 'N/A';
        }
        
        row.appendChild(paramCell);
        row.appendChild(valueCell);
        row.appendChild(confidenceCell);
        row.appendChild(deltaECell);
        
        resultsBody.appendChild(row);
    });
    
    showResults();
}

// Update file label when file is selected
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const labelText = document.querySelector('.file-label-text');
        labelText.textContent = file.name;
    }
});

