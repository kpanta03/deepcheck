// script.js
document.getElementById('uploadForm').addEventListener('submit', async function (e) {
    e.preventDefault();  // Prevent the default form submission behavior

    const formData = new FormData();
    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an audio or video file to upload.');
        return;
    }

    formData.append('audio_file', file);

    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'none';  // Hide previous result if any
    resultDiv.classList.remove('error', 'success');

    // Display a loading message
    resultDiv.innerHTML = 'Detecting audio... Please wait.';
    resultDiv.style.display = 'block';

    try {
        const response = await fetch('http://127.0.0.1:8000/api/audio/detect_audio/', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (response.ok) {
            resultDiv.classList.add('success');
            resultDiv.innerHTML = `Detection Result: ${data.result.toUpperCase()}<br>Confidence: ${data.confidence}%`;
        } else {
            resultDiv.classList.add('error');
            resultDiv.innerHTML = `Error: ${data.message}`;
        }
    } catch (error) {
        resultDiv.classList.add('error');
        resultDiv.innerHTML = `Error: Unable to process the request. Please try again later.`;
    }
});
