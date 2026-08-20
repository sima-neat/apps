document.getElementById('uploadButton').addEventListener('click', function() {
    const fileInput = document.getElementById('fileInput');
    
    // Check if a file is selected
    if (fileInput.files.length === 0) {
        alert("Please select a file.");
        return;
    }

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    // Replace 'http://yourserver.com/upload' with your server's upload URL
    fetch('http://yourserver.com/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json(); // Assuming the server responds with JSON
        })
        .then(data => {
            console.log('Success:', data);
            alert('Image uploaded successfully!');
        })
        .catch(error => {
            console.error('Error:', error);
            alert('There was an error uploading the image.');
        });
});
