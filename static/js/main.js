document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.querySelector('.drop-zone');
    const fileInput = document.querySelector('input[name="file"]');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    
    // Elements to hide/show dynamically
    const dropZoneText = document.querySelector('.drop-zone p'); 
    const dropZoneIcon = document.querySelector('.drop-zone .upload-icon, .drop-zone .fa-eye');
    const retryBtn = document.getElementById('retry-upload-btn') || document.getElementById('retry-upload-btn-eye'); 

    const form = document.getElementById('upload-form');
    const submitBtn = document.querySelector('.submit-btn');
    const btnText = document.querySelector('.btn-text');
    const spinner = document.querySelector('.spinner');

    const modal = document.getElementById('results-modal');
    const closeModalBtn = document.getElementById('close-modal-btn');


    // --- File Handling & Preview ---
    function handleFile(file) {
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                // 1. Show Image Preview
                previewImage.src = e.target.result;
                previewContainer.style.display = 'block';

                // 2. Hide Original Drop Zone (Fixing user request)
                if (dropZone) dropZone.classList.add('hidden');

                // 3. Show Retry Button (New feature)
                if (retryBtn) retryBtn.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    }

    if (dropZone && fileInput) {
        // Drag over effect
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });

        // Drag leave effect
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });

        // Drop effect
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFile(fileInput.files[0]);
            }
        });

        // Click input change
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                handleFile(fileInput.files[0]);
            }
        });
    }
    
    // --- Retry Button Logic ---
    if (retryBtn) {
        retryBtn.addEventListener('click', () => {
            fileInput.value = ''; 
            previewContainer.style.display = 'none'; 
            dropZone.classList.remove('hidden'); 
            retryBtn.style.display = 'none'; 
            
            if (dropZoneText) dropZoneText.style.display = 'block';
            if (dropZoneIcon) dropZoneIcon.style.display = 'block';
        });
    }


    // --- Form Submission Animation ---
    if (form) {
        form.addEventListener('submit', () => {
            submitBtn.disabled = true;
            btnText.textContent = 'Analyzing...';
            spinner.style.display = 'inline-block';
        });
    }

    // --- Modal Close Functionality ---
    if (modal && closeModalBtn) {
        closeModalBtn.addEventListener('click', () => {
            modal.classList.remove('active');
        });

        modal.addEventListener('click', (e) => {
            if (e.target.id === 'results-modal') {
                modal.classList.remove('active');
            }
        });
    }
});